# Argos Observability Dashboard

FastAPI serves the dashboard API and the built Vite app. Vite runs separately
during frontend development and proxies `/api` to FastAPI.

```bash
source setup_shell.sh
uvicorn argos_src.observability.dashboard_server:app --host 127.0.0.1 --port 8765 --reload
```

```bash
cd dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173` during development. After `npm run build`,
FastAPI serves the built app from `http://127.0.0.1:8765`.

By default the API reads `logs/latency.log`. Override it with:

```bash
ARGOS_DASHBOARD_LOG_PATH=/path/to/latency.log uvicorn argos_src.observability.dashboard_server:app --host 127.0.0.1 --port 8765
```
