import React from "react";
import ReactDOM from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  Database,
  Fingerprint,
  MessagesSquare,
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
  raw_session_count?: number;
  exchange_count?: number;
  conversation_segment_count?: number;
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
  first_audio_latency_max_s?: number | null;
  tool_latency_avg_s: number | null;
  total_logged_cost_usd?: number | null;
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
  session_total_cost_usd?: number | null;
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
  conversation_segment_id?: string;
  conversation_segment_index?: number;
  owner_key?: string;
  timeline: TimelineItem[];
};

type ConversationSegment = {
  segment_id: string;
  session_id: string;
  segment_index: number;
  owner_key: string;
  owner_id: string;
  owner_label: string;
  started_at: string | null;
  ended_at: string | null;
  duration_s: number | null;
  exchange_count: number;
  exchange_ids: string[];
  exchange_indexes: number[];
  first_exchange_id: string;
  latest_exchange_id: string;
  status: string;
  status_counts: Record<string, number>;
  owner_source_counts: Record<string, number>;
  owner_sources: string[];
  avg_first_audio_latency_s: number | null;
  total_exchange_cost_usd: number | null;
  handoff_from_owner_key: string;
  handoff_to_owner_key: string;
  boundary_reason: string;
};

type DashboardSnapshot = {
  source: string;
  generated_at: string;
  summary: Summary;
  sessions: Session[];
  conversation_segments?: ConversationSegment[];
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
    raw_session_count: 0,
    exchange_count: 0,
    conversation_segment_count: 0,
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
    first_audio_latency_max_s: null,
    tool_latency_avg_s: null,
    total_logged_cost_usd: null,
    latest_session_total_cost_usd: null
  },
  sessions: [],
  conversation_segments: [],
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
  owner_handoff: <MessagesSquare size={18} />,
  memory_flushed: <MessagesSquare size={18} />,
  tailwag_episode_recorded: <Database size={18} />,
  tailwag_episode_failed: <AlertTriangle size={18} />,
  tailwag_episode_skipped: <Database size={18} />,
  biometric_update: <Fingerprint size={18} />,
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

function shortSessionId(session: Session) {
  const id = session.session_id || "";
  if (!id) return "unknown";
  return id.length <= 8 ? id : id.slice(-6);
}

function formatSessionTitle(session: Session, sessions: Session[]) {
  const range = formatSessionRange(session);
  const duplicateRange = sessions.filter((candidate) => formatSessionRange(candidate) === range).length > 1;
  const suffix = duplicateRange ? ` · ${shortSessionId(session)}` : "";
  return `Run ${range}${suffix}`;
}

function median(values: number[]) {
  const clean = [...values].filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) return null;
  const middle = Math.floor(clean.length / 2);
  if (clean.length % 2) return clean[middle];
  return (clean[middle - 1] + clean[middle]) / 2;
}

function latestSessionCost(exchanges: Exchange[]) {
  const dated = exchanges
    .map((exchange) => ({
      ts: exchange.ended_at ?? exchange.started_at ?? "",
      cost: exchange.costs.session_total_cost_usd
    }))
    .filter((item): item is { ts: string; cost: number } => typeof item.cost === "number");
  dated.sort((a, b) => a.ts.localeCompare(b.ts));
  return dated.at(-1)?.cost ?? null;
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

const stageDetailKeys: Record<string, string[]> = {
  recording: [
    "trigger",
    "admission_reason",
    "interaction_state",
    "primary_face_person_id",
    "visible_face_person_ids"
  ],
  speech_end: ["capture_vad_positive_blocks"],
  audio_commit: [],
  identity: [
    "primary_face_person_id",
    "face_match_status",
    "face_match_reason",
    "face_match_name",
    "face_match_person_id",
    "face_score",
    "face_score_threshold",
    "face_runner_up_score",
    "face_score_margin",
    "face_margin_threshold",
    "audio_speaker_id",
    "audio_score",
    "audio_runner_up_score",
    "audio_score_margin",
    "owner_id",
    "owner_source",
    "speaker_visible",
    "audio_duration_s",
    "capture_vad_positive_blocks"
  ],
  owner_handoff: ["old_owner_key", "new_owner_key", "deleted_items", "protected_items", "history_action"],
  memory_flushed: [
    "memory_person_id",
    "memory_turn_count",
    "memory_flush_reason",
    "memory_extraction_scheduled"
  ],
  tailwag_episode_recorded: [
    "tailwag_episode_id",
    "tailwag_episode_extract_memory",
    "tailwag_memory_result_count",
    "tailwag_memory_created_count",
    "tailwag_memory_addressed_count",
    "tailwag_memory_supported_count",
    "tailwag_memory_error_count"
  ],
  tailwag_episode_failed: ["tailwag_episode_error"],
  tailwag_episode_skipped: [
    "memory_person_id",
    "memory_turn_count",
    "memory_flush_reason",
    "memory_extraction_enabled"
  ],
  biometric_update: [
    "biometric_update_modality",
    "biometric_update_person_id",
    "biometric_update_accepted",
    "biometric_update_status",
    "biometric_update_reason",
    "biometric_update_sample_count",
    "biometric_update_target_sample_count",
    "biometric_update_similarity",
    "biometric_update_reference_id"
  ],
  model_requested: ["pending_internal_events"],
  first_audio: ["duration_s"],
  tool_requested: ["tool", "call_id", "tool_arguments_json"],
  tool_finished: [
    "tool",
    "tool_success",
    "tool_enrollment_failure_reason",
    "tool_enrollment_accepted_frames",
    "tool_enrollment_consistent_frames",
    "tool_enrollment_required_frames",
    "tool_enrollment_similarity_threshold",
    "tool_enrollment_best_failed_similarity",
    "tool_enrollment_best_failed_shortfall",
    "tool_enrollment_similarities",
    "tool_result_preview"
  ],
  response_done: ["response_status"],
  response_usage: [
    "estimated_cost_usd",
    "session_total_cost_usd",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cache_hit_ratio"
  ],
  playback_completed: [],
  playback_stopped: ["terminal_reason"],
  exchange_complete: ["terminal_status", "terminal_reason"],
  exchange_terminal: [
    "terminal_status",
    "terminal_reason",
    "error_source",
    "error_type",
    "error_code",
    "error_message",
    "server_error_type",
    "server_error_code",
    "server_error_message"
  ]
};

function StatusPill({ status }: { status: string }) {
  if (status === "complete") return null;
  const label = status === "active" ? "in progress" : humanize(status);
  return <span className={`status status-${status}`}>{label}</span>;
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
  tone?: "warn" | "good" | "danger";
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

function stateAxisCountsForExchange(exchange?: Exchange) {
  const counts: Record<string, number> = {};
  for (const group of exchange?.state_by_axis ?? []) {
    counts[group.axis] = group.transitions.length + group.ignored.length;
  }
  return counts;
}

function Stage({ stage }: { stage: LifecycleStage }) {
  const detailKeys = stageDetailKeys[stage.key] ?? [];
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

function segmentRange(segment?: ConversationSegment) {
  if (!segment) return "n/a";
  const start = formatTime(segment.started_at);
  const end = formatTime(segment.ended_at);
  if (start === "n/a") return "n/a";
  if (end === "n/a" || end === start) return start;
  return `${start} - ${end}`;
}

function ownerSourceLabel(segment?: ConversationSegment) {
  const sources = segment?.owner_sources ?? [];
  if (!sources.length) return "unknown";
  return sources.map(humanize).join(", ");
}

function segmentMatchesQuery(segment: ConversationSegment, queryText: string) {
  if (!queryText) return true;
  const haystack = [
    segment.owner_label,
    segment.owner_id,
    segment.owner_key,
    segment.boundary_reason,
    segment.owner_sources.join(" "),
    segment.exchange_ids.join(" ")
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(queryText);
}

function exchangeMatchesQuery(exchange: Exchange, queryText: string) {
  if (!queryText) return true;
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
  return haystack.includes(queryText);
}

function ConversationSummary({ segment }: { segment?: ConversationSegment }) {
  if (!segment) return null;
  return (
    <section className={`conversation-summary ${segment.owner_id ? "resolved" : "anonymous"}`}>
      <div>
        <span>{segment.boundary_reason === "owner_handoff" ? "Owner handoff" : "Conversation segment"}</span>
        <strong>{segment.owner_label}</strong>
      </div>
      <div className="conversation-summary-stats">
        <span>{segment.exchange_count} exchanges</span>
        <span>{ownerSourceLabel(segment)}</span>
        <span>{formatNumber(segment.avg_first_audio_latency_s, "s")} avg first reply</span>
      </div>
    </section>
  );
}

function App() {
  const [snapshot, setSnapshot] = React.useState<DashboardSnapshot>(emptySnapshot);
  const [loading, setLoading] = React.useState(false);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = React.useState<string | null>(null);
  const [selectedConversationSegmentId, setSelectedConversationSegmentId] = React.useState<string | null>(null);
  const [selectedExchangeId, setSelectedExchangeId] = React.useState<string | null>(null);
  const [query, setQuery] = React.useState("");
  const [autoRefresh, setAutoRefresh] = React.useState(false);

  const loadSnapshot = React.useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/snapshot");
      if (!response.ok) throw new Error(`snapshot ${response.status}`);
      const body = (await response.json()) as DashboardSnapshot;
      const exchanges = body.exchanges ?? body.interactions ?? [];
      const conversationSegments = body.conversation_segments ?? [];
      setSnapshot({ ...body, conversation_segments: conversationSegments, exchanges });
      setApiError(null);
      setSelectedSessionId((current) =>
        current && body.sessions.some((session) => session.session_id === current)
          ? current
          : body.sessions[0]?.session_id ?? null
      );
      setSelectedConversationSegmentId((current) =>
        current && conversationSegments.some((segment) => segment.segment_id === current)
          ? current
          : conversationSegments[0]?.segment_id ?? null
      );
      setSelectedExchangeId((current) =>
        current && exchanges.some((exchange) => exchange.exchange_id === current)
          ? current
          : exchanges[0]?.exchange_id ?? null
      );
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "snapshot unavailable");
      setSnapshot({ ...emptySnapshot, generated_at: new Date().toISOString() });
      setSelectedSessionId(null);
      setSelectedConversationSegmentId(null);
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
  const conversationSegments = snapshot.conversation_segments ?? [];
  const selectedSession =
    snapshot.sessions.find((session) => session.session_id === selectedSessionId) ?? snapshot.sessions[0];
  const sessionExchanges = selectedSession
    ? exchanges.filter((exchange) => exchange.session_id === selectedSession.session_id)
    : exchanges;
  const sessionSegments = selectedSession
    ? conversationSegments.filter((segment) => segment.session_id === selectedSession.session_id)
    : conversationSegments;
  const queryText = query.toLowerCase();
  const exchangesBySegment = new Map<string, Exchange[]>();
  for (const exchange of sessionExchanges) {
    const segmentId = exchange.conversation_segment_id ?? "";
    if (!exchangesBySegment.has(segmentId)) exchangesBySegment.set(segmentId, []);
    exchangesBySegment.get(segmentId)?.push(exchange);
  }
  const filteredSegments = sessionSegments.filter((segment) => {
    const segmentExchanges = exchangesBySegment.get(segment.segment_id) ?? [];
    return segmentMatchesQuery(segment, queryText) || segmentExchanges.some((exchange) => exchangeMatchesQuery(exchange, queryText));
  });
  const visibleExchanges = filteredSegments.flatMap((segment) => {
    const segmentExchanges = exchangesBySegment.get(segment.segment_id) ?? [];
    if (segmentMatchesQuery(segment, queryText)) return segmentExchanges;
    return segmentExchanges.filter((exchange) => exchangeMatchesQuery(exchange, queryText));
  });
  const selectedConversationSegment =
    filteredSegments.find((segment) => segment.segment_id === selectedConversationSegmentId) ??
    filteredSegments[0] ??
    sessionSegments[0];
  const selectedExchange =
    visibleExchanges.find((exchange) => exchange.exchange_id === selectedExchangeId) ??
    (selectedConversationSegment
      ? (exchangesBySegment.get(selectedConversationSegment.segment_id) ?? [])[0]
      : undefined) ??
    visibleExchanges[0] ??
    sessionExchanges[0] ??
    exchanges[0];
  const activeConversationSegment =
    sessionSegments.find((segment) => segment.segment_id === selectedExchange?.conversation_segment_id) ??
    selectedConversationSegment;
  const latencyValues = sessionExchanges
    .map((exchange) => exchange.first_audio_latency_s)
    .filter((value): value is number => typeof value === "number");
  const medianFirstAudio = selectedSession
    ? median(latencyValues)
    : snapshot.summary.first_audio_latency_p50_s;
  const selectedSessionCost = selectedSession
    ? selectedSession.session_total_cost_usd ?? latestSessionCost(sessionExchanges)
    : snapshot.summary.latest_session_total_cost_usd;
  const selectedErrorCount = selectedSession
    ? selectedSession.error_count
    : snapshot.summary.error_count;
  const selectedExchangeCost =
    selectedExchange?.costs.estimated_exchange_cost_usd ?? selectedExchange?.costs.estimated_cost_usd ?? null;
  const selectedExchangeStateAxisCounts = stateAxisCountsForExchange(selectedExchange);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Bot size={28} />
          <div>
            <h1>Tailwag Observability Dashboard</h1>
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
        <MetricCard icon={<MessagesSquare size={20} />} label="Conversations" value={`${sessionSegments.length}`} />
        <MetricCard icon={<Activity size={20} />} label="Exchanges" value={`${sessionExchanges.length}`} />
        <MetricCard
          icon={<AlertTriangle size={20} />}
          label="Errors"
          value={`${selectedErrorCount}`}
          tone="danger"
        />
        <MetricCard
          icon={<Clock3 size={20} />}
          label="Median First Reply Audio"
          value={formatNumber(medianFirstAudio, "s")}
        />
        <MetricCard
          icon={<Zap size={20} />}
          label="Selected Session Cost"
          value={formatCost(selectedSessionCost)}
        />
        <MetricCard
          icon={<CircleDollarSign size={20} />}
          label="Cost To Date"
          value={formatCost(snapshot.summary.total_logged_cost_usd)}
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
                className={`session-row ${selectedSession?.session_id === session.session_id ? "selected" : ""}`}
                key={session.session_id}
                type="button"
                onClick={() => {
                  const firstSegment = conversationSegments.find((item) => item.session_id === session.session_id);
                  const first = firstSegment
                    ? exchanges.find((item) => item.conversation_segment_id === firstSegment.segment_id)
                    : exchanges.find((item) => item.session_id === session.session_id);
                  setSelectedSessionId(session.session_id);
                  setSelectedConversationSegmentId(firstSegment?.segment_id ?? null);
                  setSelectedExchangeId(first?.exchange_id ?? null);
                }}
              >
                <strong>{formatSessionTitle(session, snapshot.sessions)}</strong>
                <span>{session.exchange_count ?? session.interaction_count} exchanges</span>
                <small>{formatNumber(session.avg_first_audio_latency_s, "s")} avg first reply</small>
              </button>
            ))}
          </div>

          <div className="search-box">
            <Search size={16} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="exchange, owner, tool" />
          </div>
          <div className="interaction-list">
            {filteredSegments.map((segment) => {
              const segmentExchanges = exchangesBySegment.get(segment.segment_id) ?? [];
              const segmentSelected = activeConversationSegment?.segment_id === segment.segment_id;
              const visibleSegmentExchanges = segmentMatchesQuery(segment, queryText)
                ? segmentExchanges
                : segmentExchanges.filter((exchange) => exchangeMatchesQuery(exchange, queryText));
              return (
                <section className={`conversation-group ${segmentSelected ? "selected" : ""}`} key={segment.segment_id}>
                  <button
                    className="conversation-row"
                    type="button"
                    onClick={() => {
                      setSelectedConversationSegmentId(segment.segment_id);
                      setSelectedExchangeId(segmentExchanges[0]?.exchange_id ?? null);
                    }}
                  >
                    <div>
                      <strong>{segment.owner_label}</strong>
                      <span>
                        {segment.exchange_count} exchanges · {ownerSourceLabel(segment)}
                      </span>
                      <small>{segmentRange(segment)}</small>
                    </div>
                    <StatusPill status={segment.status} />
                  </button>
                  <div className="conversation-exchanges">
                    {visibleSegmentExchanges.map((exchange) => (
                      <button
                        className={`interaction-row ${selectedExchange?.exchange_id === exchange.exchange_id ? "selected" : ""}`}
                        key={exchange.exchange_id}
                        type="button"
                        onClick={() => {
                          setSelectedConversationSegmentId(segment.segment_id);
                          setSelectedExchangeId(exchange.exchange_id);
                        }}
                      >
                        <div>
                          <strong>{exchange.label}</strong>
                          <span>{formatTime(exchange.started_at)} human {"->"} Tailwag</span>
                        </div>
                        <StatusPill status={exchange.status} />
                      </button>
                    ))}
                  </div>
                </section>
              );
            })}
          </div>
        </aside>

        <section className="timeline-panel">
          <div className="panel-heading">
            <div>
              <h2>{selectedExchange?.label ?? "No exchange"}</h2>
              <p>{selectedExchange ? `${formatTime(selectedExchange.started_at)} human -> Tailwag` : "No exchange selected"}</p>
            </div>
            {selectedExchange ? <StatusPill status={selectedExchange.status} /> : null}
          </div>

          <ConversationSummary segment={activeConversationSegment} />

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
          </div>
        </section>

        <aside className="inspector">
          <div className="panel">
            <div className="panel-heading">
              <h2>Conversation</h2>
              <MessagesSquare size={18} />
            </div>
            <KeyValueList
              values={[
                ["owner", activeConversationSegment?.owner_label],
                ["boundary", activeConversationSegment?.boundary_reason],
                ["exchanges", activeConversationSegment?.exchange_count],
                ["owner_sources", activeConversationSegment?.owner_sources.join(", ")],
                ["avg_first_reply", formatNumber(activeConversationSegment?.avg_first_audio_latency_s, "s")],
                ["segment_cost", formatCost(activeConversationSegment?.total_exchange_cost_usd)]
              ]}
            />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <h2>Exchange Summary</h2>
              <span>{formatNumber(selectedExchange?.duration_s, "s")}</span>
            </div>
            <KeyValueList
              values={[
                ["trigger", selectedExchange?.context.trigger],
                ["admission_reason", selectedExchange?.context.admission_reason],
                ["first_reply_audio", formatNumber(selectedExchange?.first_audio_latency_s, "s")],
                ["exchange_cost", selectedExchangeCost === null ? null : formatCost(Number(selectedExchangeCost))],
                ["terminal_reason", selectedExchange?.context.terminal_reason],
                ["error_source", selectedExchange?.context.error_source],
                ["error_type", selectedExchange?.context.error_type],
                ["error_code", selectedExchange?.context.error_code],
                ["error_message", selectedExchange?.context.error_message],
                ["server_error_type", selectedExchange?.context.server_error_type],
                ["server_error_code", selectedExchange?.context.server_error_code],
                ["server_error_message", selectedExchange?.context.server_error_message],
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
                ["conversation_segment", selectedExchange?.conversation_segment_id],
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
              <BarList values={selectedExchangeStateAxisCounts} />
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
