import { spawn, type ChildProcess } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import dotenv from 'dotenv';
import express from 'express';

dotenv.config();

const PORT = Number(process.env.PORT || 3000);
const CONFIGURED_API_URL = process.env.API_URL?.replace(/\/$/, '') || '';
const RUNTIME_API_KEY = process.env.API_KEY || crypto.randomBytes(24).toString('hex');

let fastapiChild: ChildProcess | null = null;
let activeFastApiUrl = CONFIGURED_API_URL || 'http://127.0.0.1:8010';
let backendReadyPromise: Promise<void> | null = null;

async function isMallSegFastApiHealthy(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(1500) });
    if (!res.ok) return false;
    const body = (await res.json()) as Record<string, unknown>;
    return (
      typeof body.database_ok === 'boolean' &&
      typeof body.model_loaded === 'boolean' &&
      'active_run_key' in body
    );
  } catch {
    return false;
  }
}

function findFreeLocalPort(preferredPort: number): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', () => {
      const fallback = net.createServer();
      fallback.unref();
      fallback.on('error', reject);
      fallback.listen(0, '127.0.0.1', () => {
        const addr = fallback.address();
        const port = typeof addr === 'object' && addr ? addr.port : preferredPort + 1;
        fallback.close(() => resolve(port));
      });
    });
    server.listen(preferredPort, '127.0.0.1', () => {
      server.close(() => resolve(preferredPort));
    });
  });
}

function resolveUvicornBin(): string {
  const venvUvicorn = path.resolve(process.cwd(), '.venv/bin/uvicorn');
  return fs.existsSync(venvUvicorn) ? venvUvicorn : 'uvicorn';
}

async function ensureFastApiRunning(): Promise<void> {
  if (CONFIGURED_API_URL && (await isMallSegFastApiHealthy(CONFIGURED_API_URL))) {
    activeFastApiUrl = CONFIGURED_API_URL;
    return;
  }

  const preferredPort = CONFIGURED_API_URL
    ? Number(new URL(CONFIGURED_API_URL).port || 8010)
    : 8010;
  const host = CONFIGURED_API_URL ? new URL(CONFIGURED_API_URL).hostname : '127.0.0.1';
  const port = await findFreeLocalPort(preferredPort);
  activeFastApiUrl = `http://${host}:${port}`;

  fastapiChild = spawn(
    resolveUvicornBin(),
    ['mall_seg.api:app', '--host', host, '--port', String(port)],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        API_KEY: RUNTIME_API_KEY,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );

  fastapiChild.stderr?.on('data', (chunk: Buffer) => {
    const msg = chunk.toString().trim();
    if (msg) console.error(`[fastapi] ${msg}`);
  });

  for (let i = 0; i < 80; i++) {
    if (await isMallSegFastApiHealthy(activeFastApiUrl)) {
      console.log(`[bff] Verified FastAPI backend at ${activeFastApiUrl}`);
      return;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  console.warn(`[bff] FastAPI did not report healthy at ${activeFastApiUrl} within startup timeout.`);
}

function shutdownChild() {
  if (fastapiChild && !fastapiChild.killed) {
    fastapiChild.kill('SIGTERM');
    fastapiChild = null;
  }
}

process.on('SIGINT', () => {
  shutdownChild();
  process.exit(0);
});
process.on('SIGTERM', () => {
  shutdownChild();
  process.exit(0);
});
process.on('exit', shutdownChild);

async function startServer() {
  backendReadyPromise = ensureFastApiRunning();

  const app = express();
  app.use(express.json({ limit: '4mb' }));

  // Proxy /api/health -> FastAPI /health (and /live /ready)
  app.get('/api/health', async (_req, res) => {
    try {
      await backendReadyPromise;
      const upstream = await fetch(`${activeFastApiUrl}/health`, {
        signal: AbortSignal.timeout(3000),
      });
      const text = await upstream.text();
      res.status(upstream.status).type('application/json').send(text);
    } catch {
      res.status(503).json({
        status: 'degraded',
        database_ok: false,
        model_loaded: false,
        active_run_key: null,
      });
    }
  });

  app.get('/api/health/:probe', async (req, res) => {
    try {
      await backendReadyPromise;
      const probe = req.params.probe === 'live' ? 'live' : 'ready';
      const upstream = await fetch(`${activeFastApiUrl}/health/${probe}`, {
        signal: AbortSignal.timeout(3000),
      });
      const text = await upstream.text();
      res.status(upstream.status).type('application/json').send(text);
    } catch {
      res.status(503).json({ status: 'degraded' });
    }
  });

  app.get('/api/metrics', async (_req, res) => {
    try {
      await backendReadyPromise;
      const upstream = await fetch(`${activeFastApiUrl}/metrics`, {
        signal: AbortSignal.timeout(3000),
      });
      const text = await upstream.text();
      res
        .status(upstream.status)
        .type(upstream.headers.get('content-type') || 'text/plain; version=0.0.4; charset=utf-8')
        .send(text);
    } catch {
      res.status(503).type('text/plain').send('# mall_seg metrics unavailable\n');
    }
  });

  // Proxy /api/v1/* -> FastAPI /api/v1/* preserving Bearer JWT or injecting server-side X-API-Key
  app.all('/api/v1/*', async (req, res) => {
    try {
      await backendReadyPromise;
      const targetUrl = `${activeFastApiUrl}${req.originalUrl}`;
      const headers: Record<string, string> = {};
      const authHeader = req.headers.authorization;
      if (typeof authHeader === 'string' && authHeader.toLowerCase().startsWith('bearer ')) {
        headers['Authorization'] = authHeader;
      } else {
        headers['X-API-Key'] = RUNTIME_API_KEY;
      }
      const incomingReqId = req.headers['x-request-id'];
      if (typeof incomingReqId === 'string') {
        headers['X-Request-ID'] = incomingReqId;
      }

      const incomingCt = req.headers['content-type'] || '';
      let bodyToSend: string | Uint8Array | undefined;
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        if (incomingCt.includes('multipart/form-data')) {
          headers['Content-Type'] = incomingCt;
          const chunks: Buffer[] = [];
          for await (const chunk of req) {
            chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
          }
          bodyToSend = new Uint8Array(Buffer.concat(chunks));
        } else {
          headers['Content-Type'] = 'application/json';
          bodyToSend = JSON.stringify(req.body ?? {});
        }
      }

      const upstream = await fetch(targetUrl, {
        method: req.method,
        headers,
        body: bodyToSend,
        signal: AbortSignal.timeout(45000),
      });

      const contentType = upstream.headers.get('content-type') || 'application/json';
      const reqId = upstream.headers.get('x-request-id');
      const procMs = upstream.headers.get('x-process-time-ms');
      if (reqId) res.setHeader('X-Request-ID', reqId);
      if (procMs) res.setHeader('X-Process-Time-Ms', procMs);

      const buf = Buffer.from(await upstream.arrayBuffer());
      res.status(upstream.status).type(contentType).send(buf);
    } catch (err) {
      console.error('[bff] Upstream proxy error:', err);
      res.status(502).json({ detail: 'Segmentation inference backend unreachable.' });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[bff] Listening on http://0.0.0.0:${PORT} (FastAPI target ${activeFastApiUrl})`);
  });
}

startServer().catch((err) => {
  console.error('[bff] Fatal startup error:', err);
  process.exit(1);
});
