import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Mic,
  Radio,
  RefreshCw,
  Search,
  UserRound,
  Volume2,
  Wrench,
  Zap
} from "lucide-react";
import "./styles.css";

type Summary = {
  row_count: number;
  session_count: number;
  exchange_count?: number;
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
  label?: string;
  started_at: string | null;
  ended_at: string | null;
  row_count: number;
  exchange_count?: number;
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

type StateAxisGroup = {
  axis: string;
  transitions: StateTransition[];
  ignored: IgnoredStateEvent[];
};

type IgnoredStateEvent = {
  axis: string;
  trigger?: string | null;
  ignored_reason: string;
  ts?: string | null;
};

type ToolCall = {
  call_id?: string | null;
  tool?: string | null;
  requested_at?: string | null;
  finished_at?: string | null;
  arguments_json?: string | null;
  result_preview?: string | null;
  success?: string | boolean | null;
};

type LifecycleStage = {
  key: string;
  title: string;
  ts?: string | null;
  label: string;
  component: string;
  details: Record<string, string | number | boolean>;
};

type Exchange = {
  exchange_id: string;
  exchange_index: number;
  label: string;
  req_id: string;
  raw_req_ids: string[];
  session_id: string;
  openai_session_ids: string[];
  started_at: string | null;
  ended_at: string | null;
  duration_s: number | null;
  status: string;
  context: Record<string, string | number | boolean>;
  lifecycle: LifecycleStage[];
  event_counts: Record<string, number>;
  metrics: Record<string, number>;
  state_transitions: StateTransition[];
  ignored_state_events: IgnoredStateEvent[];
  state_by_axis?: StateAxisGroup[];
  tools: Record<string, number>;
  tool_calls?: ToolCall[];
  costs: Record<string, number>;
  first_audio_latency_s: number | null;
  timeline: TimelineItem[];
};

type DashboardSnapshot = {
  source: string;
  generated_at: string;
  summary: Summary;
  sessions: Session[];
  exchanges?: Exchange[];
  interactions: Exchange[];
  system_events: TimelineItem[];
  errors: TimelineItem[];
};

const emptySnapshot: DashboardSnapshot = {
  source: "logs/latency.log",
  generated_at: new Date().toISOString(),
  summary: {
    row_count: 0,
    session_count: 0,
    exchange_count: 0,
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
  exchanges: [],
  interactions: [],
  system_events: [],
  errors: []
};

const stageIcon: Record<string, React.ReactNode> = {
  recording: <Mic size={18} />,
  speech_end: <Mic size={18} />,
  audio_commit: <Radio size={18} />,
  identity: <UserRound size={18} />,
  model_requested: <Bot size={18} />,
  first_audio: <Volume2 size={18} />,
  tool_requested: <Wrench size={18} />,
  tool_finished: <Wrench size={18} />,
  response_done: <CheckCircle2 size={18} />,
  response_usage: <Zap size={18} />,
  playback_completed: <Volume2 size={18} />,
  playback_stopped: <AlertTriangle size={18} />,
  exchange_complete: <CheckCircle2 size={18} />,
  exchange_terminal: <AlertTriangle size={18} />
};

function formatNumber(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(value >= 10 ? 1 : 3)}${suffix}`;
}

function formatCost(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `$${value.toFixed(4)}`;
}

function formatTime(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatSessionRange(session: Session) {
  const start = formatTime(session.started_at);
  const end = formatTime(session.ended_at);
  if (start === "n/a") return session.label ?? "Unknown session";
  if (end === "n/a" || end === start) return start;
  return `${start} - ${end}`;
}

function humanize(value: string) {
  return value.replaceAll("_", " ");
}

function sortedEntries(record: Record<string, number>) {
  return Object.entries(record).sort((a, b) => b[1] - a[1]);
}

function compactDetails(details: Record<string, string | number | boolean>, keys: string[]) {
  return keys
    .map((key) => [key, details[key]] as const)
    .filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status status-${status}`}>{humanize(status)}</span>;
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

function KeyValueList({ values }: { values: Array<[string, string | number | boolean | null | undefined]> }) {
  const visible = values.filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (visible.length === 0) return <span className="muted">none</span>;
  return (
    <dl className="detail-grid">
      {visible.map(([label, value]) => (
        <React.Fragment key={label}>
          <dt>{humanize(label)}</dt>
          <dd>{String(value)}</dd>
        </React.Fragment>
      ))}
    </dl>
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
          <span>{humanize(label)}</span>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.max((count / max) * 100, 6)}%` }} />
          </div>
          <strong>{count}</strong>
        </div>
      ))}
    </div>
  );
}

function Stage({ stage }: { stage: LifecycleStage }) {
  const detailKeys = [
    "trigger",
    "admission_reason",
    "interaction_state",
    "primary_face_person_id",
    "visible_face_person_ids",
    "audio_speaker_id",
    "owner_id",
    "owner_source",
    "owner_confidence",
    "speaker_visible",
    "tool",
    "response_status",
    "terminal_status",
    "terminal_reason",
    "duration_s",
    "audio_duration_s",
    "capture_vad_positive_blocks"
  ];
  const details = compactDetails(stage.details, detailKeys);
  return (
    <div className={`stage stage-${stage.key}`}>
      <div className="stage-icon">{stageIcon[stage.key] ?? <Activity size={18} />}</div>
      <div className="stage-body">
        <div className="stage-title">
          <strong>{stage.title}</strong>
          <span>{formatTime(stage.ts)}</span>
        </div>
        {details.length ? (
          <div className="stage-details">
            {details.map(([key, value]) => (
              <span key={key}>
                {humanize(key)}: <b>{String(value)}</b>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function StateAxisCard({ group }: { group: StateAxisGroup }) {
  const latest = [...group.transitions].reverse().find((item) => item.new_state);
  return (
    <section className="axis-card">
      <div className="axis-card-head">
        <strong>{humanize(group.axis)}</strong>
        <span>{latest?.new_state ? humanize(latest.new_state) : `${group.transitions.length} transitions`}</span>
      </div>
      <div className="axis-events">
        {group.transitions.slice(-6).map((item, index) => (
          <div className="axis-event" key={`${item.ts}-${item.trigger}-${index}`}>
            <span>{formatTime(item.ts)}</span>
            <b>
              {humanize(item.old_state ?? "?")} {"->"} {humanize(item.new_state ?? "?")}
            </b>
            {item.trigger ? <small>{humanize(item.trigger)}</small> : null}
          </div>
        ))}
        {group.ignored.slice(-3).map((item, index) => (
          <div className="axis-event ignored" key={`ignored-${item.ts}-${index}`}>
            <span>{formatTime(item.ts)}</span>
            <b>{humanize(item.trigger ?? "ignored")}</b>
            <small>{humanize(item.ignored_reason)}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

function ToolCallList({ calls }: { calls: ToolCall[] }) {
  if (!calls.length) return <span className="muted">none</span>;
  return (
    <div className="tool-call-list">
      {calls.map((call, index) => (
        <section className="tool-call" key={`${call.call_id ?? call.tool ?? "tool"}-${index}`}>
          <div className="tool-call-head">
            <strong>{call.tool ? humanize(call.tool) : "tool call"}</strong>
            <span>{call.success === undefined || call.success === null ? "unknown" : String(call.success)}</span>
          </div>
          <KeyValueList
            values={[
              ["call_id", call.call_id],
              ["requested", formatTime(call.requested_at)],
              ["finished", formatTime(call.finished_at)]
            ]}
          />
          {call.arguments_json ? (
            <div className="code-preview">
              <span>arguments</span>
              <code>{call.arguments_json}</code>
            </div>
          ) : null}
          {call.result_preview ? (
            <div className="code-preview">
              <span>result</span>
              <code>{call.result_preview}</code>
            </div>
          ) : null}
        </section>
      ))}
    </div>
  );
}

function OwnerResolution({ exchange }: { exchange?: Exchange }) {
  const context = exchange?.context ?? {};
  const owner = context.owner_id;
  return (
    <div className={`owner-resolution ${owner ? "resolved" : "unresolved"}`}>
      <strong>{owner ? `Owner: ${String(owner)}` : "No resolved owner"}</strong>
      <span>
        {owner
          ? `source ${String(context.owner_source ?? "unknown")}`
          : "No owner_id was logged for this exchange. Check face, speaker, and owner_source evidence below."}
      </span>
    </div>
  );
}

function App() {
  const [snapshot, setSnapshot] = React.useState<DashboardSnapshot>(emptySnapshot);
  const [loading, setLoading] = React.useState(false);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [selectedExchangeId, setSelectedExchangeId] = React.useState<string | null>(null);
  const [query, setQuery] = React.useState("");
  const [statusFilter, setStatusFilter] = React.useState("all");
  const [autoRefresh, setAutoRefresh] = React.useState(false);

  const loadSnapshot = React.useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/snapshot");
      if (!response.ok) throw new Error(`snapshot ${response.status}`);
      const body = (await response.json()) as DashboardSnapshot;
      const exchanges = body.exchanges ?? body.interactions ?? [];
      setSnapshot({ ...body, exchanges });
      setApiError(null);
      setSelectedExchangeId((current) => current ?? exchanges[0]?.exchange_id ?? null);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "snapshot unavailable");
      setSnapshot({ ...emptySnapshot, generated_at: new Date().toISOString() });
      setSelectedExchangeId(null);
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

  const exchanges = snapshot.exchanges ?? snapshot.interactions ?? [];
  const filteredExchanges = exchanges.filter((exchange) => {
    const matchesStatus = statusFilter === "all" || exchange.status === statusFilter;
    const haystack = [
      exchange.label,
      exchange.exchange_id,
      exchange.req_id,
      exchange.session_id,
      exchange.context.owner_id,
      exchange.context.audio_speaker_id,
      exchange.context.primary_face_person_id,
      Object.keys(exchange.tools).join(" ")
    ]
      .join(" ")
      .toLowerCase();
    return matchesStatus && haystack.includes(query.toLowerCase());
  });
  const selectedExchange =
    filteredExchanges.find((exchange) => exchange.exchange_id === selectedExchangeId) ??
    filteredExchanges[0] ??
    exchanges[0];
  const exchangeCount = snapshot.summary.exchange_count ?? snapshot.summary.interaction_count;
  const selectedCost =
    selectedExchange?.costs.estimated_cost_usd ?? selectedExchange?.costs.session_total_cost_usd ?? null;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Bot size={28} />
          <div>
            <h1>Argos Exchange Dashboard</h1>
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
        <MetricCard icon={<Radio size={20} />} label="Sessions" value={`${snapshot.summary.session_count}`} />
        <MetricCard icon={<Activity size={20} />} label="Exchanges" value={`${exchangeCount}`} />
        <MetricCard
          icon={<AlertTriangle size={20} />}
          label="Errors"
          value={`${snapshot.summary.error_count}`}
          tone={snapshot.summary.error_count > 0 ? "warn" : "good"}
        />
        <MetricCard
          icon={<Clock3 size={20} />}
          label="Median First Audio"
          value={formatNumber(snapshot.summary.first_audio_latency_p50_s, "s")}
        />
        <MetricCard
          icon={<Volume2 size={20} />}
          label="Slowest First Audio"
          value={formatNumber(snapshot.summary.first_audio_latency_p95_s, "s")}
        />
        <MetricCard
          icon={<Zap size={20} />}
          label="Estimated Cost"
          value={formatCost(snapshot.summary.latest_session_total_cost_usd)}
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
                  const first = exchanges.find((item) => item.session_id === session.session_id);
                  setSelectedExchangeId(first?.exchange_id ?? null);
                }}
              >
                <strong>{formatSessionRange(session)}</strong>
                <span>{session.exchange_count ?? session.interaction_count} exchanges</span>
                <small>{formatNumber(session.avg_first_audio_latency_s, "s")} avg first audio</small>
              </button>
            ))}
          </div>

          <div className="search-box">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="exchange, owner, tool" />
          </div>
          <div className="segmented">
            {["all", "complete", "active", "error"].map((status) => (
              <button
                key={status}
                className={statusFilter === status ? "active" : ""}
                type="button"
                onClick={() => setStatusFilter(status)}
              >
                {humanize(status)}
              </button>
            ))}
          </div>
          <div className="interaction-list">
            {filteredExchanges.map((exchange) => (
              <button
                className={`interaction-row ${selectedExchange?.exchange_id === exchange.exchange_id ? "selected" : ""}`}
                key={exchange.exchange_id}
                type="button"
                onClick={() => setSelectedExchangeId(exchange.exchange_id)}
              >
                <div>
                  <strong>{exchange.label}</strong>
                  <span>{formatTime(exchange.started_at)} human {"->"} Argos</span>
                </div>
                <StatusPill status={exchange.status} />
              </button>
            ))}
          </div>
        </aside>

        <section className="timeline-panel">
          <div className="panel-heading">
            <div>
              <h2>{selectedExchange?.label ?? "No exchange"}</h2>
              <p>{selectedExchange ? `${formatTime(selectedExchange.started_at)} human -> Argos` : "No exchange selected"}</p>
            </div>
            {selectedExchange ? <StatusPill status={selectedExchange.status} /> : null}
          </div>

          <div className="flow-grid">
            <div className="timeline">
            {selectedExchange ? null : (
              <div className="empty-state">
                <Activity size={24} />
                <strong>No exchanges loaded</strong>
                <span>No dashboard exchange rows were found in the configured latency log.</span>
              </div>
            )}
            {(selectedExchange?.lifecycle ?? []).map((stage, index) => (
              <Stage stage={stage} key={`${stage.key}-${stage.ts ?? index}`} />
            ))}
            </div>
            <div className="axis-flow">
              <div className="axis-flow-head">
                <strong>State trajectory</strong>
                <span>{selectedExchange?.state_transitions.length ?? 0} transitions</span>
              </div>
              {(selectedExchange?.state_by_axis ?? []).length ? null : (
                <span className="muted">no state transitions</span>
              )}
              {(selectedExchange?.state_by_axis ?? []).map((group) => (
                <StateAxisCard group={group} key={group.axis} />
              ))}
            </div>
          </div>
        </section>

        <aside className="inspector">
          <div className="panel">
            <div className="panel-heading">
              <h2>Exchange Summary</h2>
              <span>{formatNumber(selectedExchange?.duration_s, "s")}</span>
            </div>
            <KeyValueList
              values={[
                ["trigger", selectedExchange?.context.trigger],
                ["admission_reason", selectedExchange?.context.admission_reason],
                ["first_audio", formatNumber(selectedExchange?.first_audio_latency_s, "s")],
                ["cost", selectedCost === null ? null : formatCost(Number(selectedCost))],
                ["terminal_reason", selectedExchange?.context.terminal_reason],
                ["tools", Object.keys(selectedExchange?.tools ?? {}).join(", ") || null]
              ]}
            />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>People Context</h2>
              <UserRound size={18} />
            </div>
            <OwnerResolution exchange={selectedExchange} />
            <KeyValueList
              values={[
                ["primary_face", selectedExchange?.context.primary_face_person_id],
                ["visible_faces", selectedExchange?.context.visible_face_person_ids],
                ["speaker", selectedExchange?.context.audio_speaker_id],
                ["owner", selectedExchange?.context.owner_id],
                ["owner_source", selectedExchange?.context.owner_source],
                ["owner_confidence", selectedExchange?.context.owner_confidence],
                ["speaker_visible", selectedExchange?.context.speaker_visible]
              ]}
            />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>Tool Calls</h2>
              <Wrench size={18} />
            </div>
            <ToolCallList calls={selectedExchange?.tool_calls ?? []} />
          </div>

          <details className="panel diagnostic-panel">
            <summary>
              <span>Diagnostics</span>
              <ChevronDown size={18} />
            </summary>
            <KeyValueList
              values={[
                ["exchange_id", selectedExchange?.exchange_id],
                ["req_id", selectedExchange?.req_id],
                ["session", selectedExchange?.session_id],
                ["openai_session", selectedExchange?.openai_session_ids?.join(", ")],
                ["state_transitions", selectedExchange?.state_transitions.length],
                ["ignored_events", selectedExchange?.ignored_state_events.length],
                ["raw_rows", selectedExchange?.timeline.length]
              ]}
            />
            <div className="diagnostic-section">
              <h3>State axes</h3>
              <BarList values={snapshot.summary.state_axis_counts} />
            </div>
            <div className="diagnostic-section">
              <h3>Raw lifecycle rows</h3>
              {(selectedExchange?.timeline ?? []).slice(0, 20).map((item) => (
                <div className="raw-row" key={`${item.index}-${item.label}`}>
                  <b>{item.label}</b>
                  <span>{item.component}</span>
                  {item.axis ? <span>{item.axis}</span> : null}
                  {item.old_state || item.new_state ? (
                    <span>
                      {item.old_state ?? "?"} {"->"} {item.new_state ?? "?"}
                    </span>
                  ) : null}
                </div>
              ))}
            </div>
          </details>
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
