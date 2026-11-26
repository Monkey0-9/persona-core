"""CLI for model registry governance: register candidate, compare, promote, and rollback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mall_seg.config import load_config
from mall_seg.db import create_db_engine, get_session_factory, init_db, session_scope
from mall_seg.registry import (
    activate_or_rollback_version,
    compare_model_versions,
    list_model_versions,
    register_candidate_model,
    serialize_model_version,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mall Segmentation Model Registry CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered model versions.")

    reg_p = sub.add_parser("register", help="Train and register a candidate model version.")
    reg_p.add_argument("--version", required=True, help="Candidate version, e.g. v1.1.0")
    reg_p.add_argument("--target-k", type=int, default=None)
    reg_p.add_argument("--include-gender", action="store_true")
    reg_p.add_argument("--notes", type=str, default=None)

    cmp_p = sub.add_parser("compare", help="Compare candidate vs active baseline.")
    cmp_p.add_argument("--version", required=True)
    cmp_p.add_argument("--baseline", default=None)

    prom_p = sub.add_parser("promote", help="Promote candidate version to active.")
    prom_p.add_argument("--version", required=True)

    rb_p = sub.add_parser("rollback", help="Roll back active model to previous/target version.")
    rb_p.add_argument("--version", default=None)

    args = parser.parse_args()
    cfg = load_config()
    engine = create_db_engine()
    init_db(engine)
    factory = get_session_factory(engine)
    default_bundle = Path(cfg.artifacts.model_bundle_path)

    if args.command == "list":
        with session_scope(factory) as session:
            items = [serialize_model_version(v) for v in list_model_versions(session)]
        print(json.dumps(items, indent=2))
        return

    if args.command == "register":
        res = register_candidate_model(
            version=args.version,
            cfg=cfg,
            engine=engine,
            bundle_path=default_bundle,
            target_k=args.target_k,
            include_gender=True if args.include_gender else None,
            notes=args.notes,
        )
        print(
            json.dumps(
                {k: v for k, v in res.items() if k not in {"bundle", "summary_payload"}},
                indent=2,
            )
        )
        return

    if args.command == "compare":
        with session_scope(factory) as session:
            cmp_res = compare_model_versions(
                session,
                candidate_version=args.version,
                baseline_version=args.baseline,
                default_bundle_path=default_bundle,
            )
        print(json.dumps(cmp_res, indent=2))
        return

    if args.command == "promote":
        with session_scope(factory) as session:
            mv, _ = activate_or_rollback_version(
                session,
                target_version=args.version,
                default_bundle_path=default_bundle,
                rollback=False,
            )
        print(json.dumps(serialize_model_version(mv), indent=2))
        return

    if args.command == "rollback":
        with session_scope(factory) as session:
            mv, _ = activate_or_rollback_version(
                session,
                target_version=args.version,
                default_bundle_path=default_bundle,
                rollback=True,
            )
        print(json.dumps(serialize_model_version(mv), indent=2))


if __name__ == "__main__":
    main()
