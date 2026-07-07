import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bot,
  Clock3,
  Gauge,
  Layers3,
  RefreshCw,
  Route,
  Search,
  Timer,
  Zap
} from "lucide-react";
import "./styles.css";

type Summary = {
  row_count: number;
  session_count: number;
  interaction_count: number;
  system_event_count: number;
  error_count: number;
  status_counts: Record<string, number>;
  component_counts: Record<string, number>;
  state_axis_counts: Record<string, number>;
  ignored_reason_counts: Record<string, number>;
  first_audio_latency_avg_s: number | null;
  first_audio_latency_p50_s: number | null;
  first_audio_latency_p95_s: number | null;
  tool_latency_avg_s: number | null;
  latest_session_total_cost_usd: number | null;
};

type Session = {
  session_id: string;
  started_at: string | null;
  ended_at: string | null;
  row_count: number;
  interaction_count: number;
  error_count: number;
  avg_first_audio_latency_s: number | null;
};

type TimelineItem = {
  index: number;
  ts?: string;
  component: string;
  kind: "event" | "metric";
  label: string;
  axis?: string | null;
  old_state?: string | null;
  new_state?: string | null;
  trigger?: string | null;
  duration_s?: number | null;
  tool?: string | null;
  ignored_reason?: string | null;
};

type StateTransition = {
  axis: string;
  old_state?: string | null;
  new_state?: string | null;
  trigger?: string | null;
  ts?: string | null;
};

type IgnoredStateEvent = {
  axis: string;
  trigger?: string | null;
  ignored_reason: string;
  ts?: string | null;
};

type Interaction = {
  req_id: string;
  session_id: string;
  started_at: string | null;
  ended_at: string | null;
  duration_s: number | null;
  status: string;
  event_counts: Record<string, number>;
  metrics: Record<string, number>;
  state_transitions: StateTransition[];
  ignored_state_events: IgnoredStateEvent[];
  tools: Record<string, number>;
  costs: Record<string, number>;
  first_audio_latency_s: number | null;
  timeline: TimelineItem[];
};

type DashboardSnapshot = {
  source: string;
  generated_at: string;
  summary: Summary;
  sessions: Session[];
  interactions: Interaction[];
  system_events: TimelineItem[];
  errors: TimelineItem[];
};

const emptySnapshot: DashboardSnapshot = {
  source: "logs/latency.log",
  generated_at: new Date().toISOString(),
  summary: {
    row_count: 0,
    session_count: 0,
    interaction_count: 0,
    system_event_count: 0,
    error_count: 0,
    status_counts: {},
    component_counts: {},
    state_axis_counts: {},
    ignored_reason_counts: {},
    first_audio_latency_avg_s: null,
    first_audio_latency_p50_s: null,
    first_audio_latency_p95_s: null,
    tool_latency_avg_s: null,
    latest_session_total_cost_usd: null
  },
  sessions: [],
  interactions: [],
  system_events: [],
  errors: []
};

function formatNumber(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(value >= 10 ? 1 : 3)}${suffix}`;
}

function formatTime(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function sortedEntries(record: Record<string, number>) {
  return Object.entries(record).sort((a, b) => b[1] - a[1]);
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

function MetricCard({
  icon,
  label,
  value,
  tone
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone?: "warn" | "good";
}) {
  return (
    <section className={`metric-card ${tone ?? ""}`}>
      <div className="metric-icon">{icon}</div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
      </div>
    </section>
  );
}

function BarList({ values }: { values: Record<string, number> }) {
  const entries = sortedEntries(values).slice(0, 8);
  const max = Math.max(...entries.map(([, count]) => count), 1);
  return (
    <div className="bar-list">
      {entries.length === 0 ? <span className="muted">none</span> : null}
      {entries.map(([label, count]) => (
        <div className="bar-row" key={label}>
          <span>{label}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.max((count / max) * 100, 6)}%` }} />
          </div>
          <strong>{count}</strong>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [snapshot, setSnapshot] = React.useState<DashboardSnapshot>(emptySnapshot);
  const [loading, setLoading] = React.useState(false);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [selectedReqId, setSelectedReqId] = React.useState<string | null>(null);
  const [query, setQuery] = React.useState("");
  const [statusFilter, setStatusFilter] = React.useState("all");
  const [autoRefresh, setAutoRefresh] = React.useState(false);

  const loadSnapshot = React.useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/snapshot");
      if (!response.ok) throw new Error(`snapshot ${response.status}`);
      const body = (await response.json()) as DashboardSnapshot;
      setSnapshot(body);
      setApiError(null);
      setSelectedReqId((current) => current ?? body.interactions[0]?.req_id ?? null);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "snapshot unavailable");
      setSnapshot({ ...emptySnapshot, generated_at: new Date().toISOString() });
      setSelectedReqId(null);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    void loadSnapshot();
  }, [loadSnapshot]);

  React.useEffect(() => {
    if (!autoRefresh) return;
    const id = window.setInterval(() => void loadSnapshot(), 5000);
    return () => window.clearInterval(id);
  }, [autoRefresh, loadSnapshot]);

  const filteredInteractions = snapshot.interactions.filter((interaction) => {
    const matchesStatus = statusFilter === "all" || interaction.status === statusFilter;
    const haystack = `${interaction.req_id} ${interaction.session_id} ${Object.keys(interaction.event_counts).join(" ")}`.toLowerCase();
    return matchesStatus && haystack.includes(query.toLowerCase());
  });
  const selectedInteraction =
    filteredInteractions.find((interaction) => interaction.req_id === selectedReqId) ??
    filteredInteractions[0] ??
    snapshot.interactions[0];

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Bot size={28} />
          <div>
            <h1>Argos Runtime Observability</h1>
            <p>{snapshot.source}</p>
          </div>
        </div>
        <div className="actions">
          <label className="toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.currentTarget.checked)}
            />
            <span>Auto</span>
          </label>
          <button className="icon-button" type="button" onClick={() => void loadSnapshot()} title="Refresh">
            <RefreshCw size={18} className={loading ? "spin" : ""} />
          </button>
        </div>
      </header>

      {apiError ? <div className="api-banner">API fallback: {apiError}</div> : null}

      <section className="metric-grid">
        <MetricCard icon={<Layers3 size={20} />} label="Sessions" value={`${snapshot.summary.session_count}`} />
        <MetricCard icon={<Activity size={20} />} label="Interactions" value={`${snapshot.summary.interaction_count}`} />
        <MetricCard
          icon={<AlertTriangle size={20} />}
          label="Errors"
          value={`${snapshot.summary.error_count}`}
          tone={snapshot.summary.error_count > 0 ? "warn" : "good"}
        />
        <MetricCard
          icon={<Clock3 size={20} />}
          label="First Audio P50"
          value={formatNumber(snapshot.summary.first_audio_latency_p50_s, "s")}
        />
        <MetricCard
          icon={<Timer size={20} />}
          label="First Audio P95"
          value={formatNumber(snapshot.summary.first_audio_latency_p95_s, "s")}
        />
        <MetricCard
          icon={<Zap size={20} />}
          label="Session Cost"
          value={
            snapshot.summary.latest_session_total_cost_usd === null
              ? "n/a"
              : `$${snapshot.summary.latest_session_total_cost_usd.toFixed(4)}`
          }
        />
      </section>

      <section className="workspace">
        <aside className="sidebar">
          <div className="panel-heading">
            <h2>Sessions</h2>
            <span>{snapshot.sessions.length}</span>
          </div>
          <div className="session-list">
            {snapshot.sessions.map((session) => (
              <button
                className="session-row"
                key={session.session_id}
                type="button"
                onClick={() => {
                  const first = snapshot.interactions.find((item) => item.session_id === session.session_id);
                  setSelectedReqId(first?.req_id ?? null);
                }}
              >
                <strong>{session.session_id}</strong>
                <span>{session.interaction_count} turns</span>
                <small>{formatNumber(session.avg_first_audio_latency_s, "s")} avg first audio</small>
              </button>
            ))}
          </div>

          <div className="search-box">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="req, session, event" />
          </div>
          <div className="segmented">
            {["all", "complete", "active", "error"].map((status) => (
              <button
                key={status}
                className={statusFilter === status ? "active" : ""}
                type="button"
                onClick={() => setStatusFilter(status)}
              >
                {status}
              </button>
            ))}
          </div>
          <div className="interaction-list">
            {filteredInteractions.map((interaction) => (
              <button
                className={`interaction-row ${selectedInteraction?.req_id === interaction.req_id ? "selected" : ""}`}
                key={interaction.req_id}
                type="button"
                onClick={() => setSelectedReqId(interaction.req_id)}
              >
                <div>
                  <strong>{interaction.req_id}</strong>
                  <span>{formatTime(interaction.started_at)}</span>
                </div>
                <StatusPill status={interaction.status} />
              </button>
            ))}
          </div>
        </aside>

        <section className="timeline-panel">
          <div className="panel-heading">
            <div>
              <h2>{selectedInteraction?.req_id ?? "No interaction"}</h2>
              <p>{selectedInteraction?.session_id ?? "No session"}</p>
            </div>
            {selectedInteraction ? <StatusPill status={selectedInteraction.status} /> : null}
          </div>

          <div className="timeline">
            {selectedInteraction ? null : (
              <div className="empty-state">
                <Activity size={24} />
                <strong>No interactions loaded</strong>
                <span>Start the agent or point `ARGOS_DASHBOARD_LOG_PATH` at a latency log.</span>
              </div>
            )}
            {(selectedInteraction?.timeline ?? []).map((item) => (
              <div className={`timeline-item component-${item.component}`} key={`${item.index}-${item.label}`}>
                <div className="timeline-rail">
                  <span />
                </div>
                <div className="timeline-content">
                  <div className="timeline-title">
                    <strong>{item.label}</strong>
                    <span>{formatTime(item.ts)}</span>
                  </div>
                  <div className="timeline-meta">
                    <span>{item.component}</span>
                    {item.axis ? <span>{item.axis}</span> : null}
                    {item.old_state || item.new_state ? (
                      <span>
                        {item.old_state ?? "?"}
                        {" -> "}
                        {item.new_state ?? "?"}
                      </span>
                    ) : null}
                    {item.duration_s !== null && item.duration_s !== undefined ? (
                      <span>{formatNumber(item.duration_s, "s")}</span>
                    ) : null}
                    {item.tool ? <span>{item.tool}</span> : null}
                    {item.ignored_reason ? <span>{item.ignored_reason}</span> : null}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <aside className="inspector">
          <div className="panel">
            <div className="panel-heading">
              <h2>Runtime Shape</h2>
              <Gauge size={18} />
            </div>
            <BarList values={snapshot.summary.component_counts} />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>State Axes</h2>
              <Route size={18} />
            </div>
            <BarList values={snapshot.summary.state_axis_counts} />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>Selected Turn</h2>
              <span>{formatNumber(selectedInteraction?.duration_s, "s")}</span>
            </div>
            <dl className="detail-grid">
              <dt>First audio</dt>
              <dd>{formatNumber(selectedInteraction?.first_audio_latency_s, "s")}</dd>
              <dt>Transitions</dt>
              <dd>{selectedInteraction?.state_transitions.length ?? 0}</dd>
              <dt>Ignored</dt>
              <dd>{selectedInteraction?.ignored_state_events.length ?? 0}</dd>
              <dt>Tools</dt>
              <dd>{Object.keys(selectedInteraction?.tools ?? {}).join(", ") || "none"}</dd>
            </dl>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>Ignored Reasons</h2>
              <AlertTriangle size={18} />
            </div>
            <BarList values={snapshot.summary.ignored_reason_counts} />
          </div>
        </aside>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
