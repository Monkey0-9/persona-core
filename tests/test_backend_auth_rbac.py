"""Integration tests for JWT auth, refresh rotation, RBAC, extra=forbid, and health probes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from mall_seg.api import create_app


@pytest.mark.anyio
async def test_jwt_rbac_extra_forbid_and_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "rbac_test.db"
    bundle_path = tmp_path / "bundle.joblib"
    monkeypatch.setenv("API_KEY", "test-bff-key")
    monkeypatch.setenv("JWT_SECRET", "unit-test-super-secret-jwt-key-123456")
    monkeypatch.setenv("MALL_SEG_AUTO_BOOTSTRAP", "1")
    monkeypatch.setenv("MALL_SEG_DISABLE_RATE_LIMIT", "1")

    app = create_app(database_url=f"sqlite:///{db_path}", bundle_path=bundle_path)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        # 1. Health liveness & readiness probes
        live = await client.get("/health/live")
        assert live.status_code == 200
        assert live.json()["status"] == "alive"

        ready = await client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ok"

        # 2. Bootstrap initial admin
        boot = await client.post(
            "/api/v1/auth/bootstrap-admin",
            json={"email": "admin@mall.local", "password": "StrongAdminPassword1!"},
        )
        assert boot.status_code == 200
        admin_tokens = boot.json()
        assert admin_tokens["user"]["role"] == "admin"

        # Second bootstrap attempt blocked with 409
        boot_dup = await client.post(
            "/api/v1/auth/bootstrap-admin",
            json={"email": "hacker@mall.local", "password": "StrongAdminPassword1!"},
        )
        assert boot_dup.status_code == 409

        # 3. Create viewer user as admin, verify RBAC blocks write scoring for viewer
        create_viewer = await client.post(
            "/api/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
            json={
                "email": "viewer@mall.local",
                "password": "ViewerPassword123!",
                "role": "viewer",
            },
        )
        assert create_viewer.status_code == 200

        viewer_login = await client.post(
            "/api/v1/auth/login",
            json={"email": "viewer@mall.local", "password": "ViewerPassword123!"},
        )
        assert viewer_login.status_code == 200
        viewer_access = viewer_login.json()["access_token"]
        viewer_refresh = viewer_login.json()["refresh_token"]

        # Viewer can read segments & active model
        seg_res = await client.get(
            "/api/v1/segments",
            headers={"Authorization": f"Bearer {viewer_access}"},
        )
        assert seg_res.status_code == 200

        model_res = await client.get(
            "/api/v1/models/active",
            headers={"Authorization": f"Bearer {viewer_access}"},
        )
        assert model_res.status_code == 200
        assert model_res.json()["version"] == "v1.0.0"

        # Viewer CANNOT call /api/v1/customers/score -> 403 Forbidden
        forbidden_score = await client.post(
            "/api/v1/customers/score",
            headers={"Authorization": f"Bearer {viewer_access}"},
            json={"gender": "Female", "age": 31, "annual_income_k": 88.0, "spending_score": 84.0},
        )
        assert forbidden_score.status_code == 403

        # 4. Admin can score customer, and extra="forbid" rejects unexpected fields with 422
        bad_extra = await client.post(
            "/api/v1/customers/score",
            headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
            json={
                "gender": "Female",
                "age": 31,
                "annual_income_k": 88.0,
                "spending_score": 84.0,
                "unexpected_admin_flag": True,
            },
        )
        assert bad_extra.status_code == 422

        good_score = await client.post(
            "/api/v1/customers/score",
            headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
            json={"gender": "Female", "age": 31, "annual_income_k": 88.0, "spending_score": 84.0},
        )
        assert good_score.status_code == 200
        s_body = good_score.json()
        assert s_body["distance_to_centroid"] >= 0.0
        assert isinstance(s_body["drift_flag"], bool)

        # 5. Drift monitoring endpoint
        drift_mon = await client.get(
            "/api/v1/monitoring/drift",
            headers={"Authorization": f"Bearer {admin_tokens['access_token']}"},
        )
        assert drift_mon.status_code == 200
        assert "reference_stats" in drift_mon.json()

        # 6. Refresh token rotation invalidates old refresh token
        refreshed = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": viewer_refresh},
        )
        assert refreshed.status_code == 200
        replay = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": viewer_refresh},
        )
        assert replay.status_code == 401
