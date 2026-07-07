# Argos Exchange Dashboard

FastAPI serves the dashboard API and the built Vite app. Vite runs separately
during frontend development and proxies `/api` to FastAPI.

The primary dashboard unit is an exchange: one admitted human speech input,
its committed Realtime turn, the model response, optional tool calls, playback,
and terminal status. Raw state-axis and component-count rows are still available
under Diagnostics, but they are not the default operator view.

The primary session list includes only operator runs that contain at least one
exchange. Startup/shutdown-only OpenAI session rows remain in the raw log and
API summary as `raw_session_count`, but they do not create main dashboard
session cards.

Headline exchange, error, first-reply-audio, and cost metrics are scoped to the
selected session. The cost card is the selected session's latest cumulative
`session_total_cost_usd`; the exchange summary shows that exchange's summed
logged `estimated_cost_usd` rows.

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
