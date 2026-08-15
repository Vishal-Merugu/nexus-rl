/**
 * Wire types — the shape the simulator's WebSocket sends.
 *
 * These mirror the pydantic models in `packages/nexus-types`. They are duplicated rather than generated
 * because the surface is small and stable; if it grows, generate them from the FastAPI OpenAPI schema
 * rather than letting the two drift by hand.
 */

export type TrafficClass = 'LATENCY_SENSITIVE' | 'BULK' | 'BACKGROUND';

export interface LinkState {
  link_id: string;
  source: string;
  target: string;
  capacity_mbps: number;
  effective_capacity_mbps: number;
  /** Total carried: newly admitted traffic plus backlog drained. */
  allocated_mbps: number;
  /** The backlog-clearing portion of `allocated_mbps`. Not attributed to any flow. */
  drained_mbps: number;
  queue_mb: number;
  /** Always in [0, 1]. Demand above capacity shows as queue growth, not as utilization > 1. */
  utilization: number;
  latency_ms: number;
}

export interface FlowState {
  flow_id: string;
  source: string;
  target: string;
  traffic_class: TrafficClass;
  demand_mbps: number;
  allocated_mbps: number;
  dropped_mbps: number;
  path: string[];
  latency_ms: number;
}

export interface NetworkState {
  /** The join key. Every panel renders the same timestep — never mix a fresh value into a stale view. */
  timestep: number;
  links: LinkState[];
  flows: FlowState[];
  active_incident_ids: string[];
}

export interface StepMetrics {
  p95_latency_ms: number;
  mean_latency_ms: number;
  total_demand_mbps: number;
  total_throughput_mbps: number;
  total_dropped_mbps: number;
  fairness_index: number;
  mean_utilization: number;
  max_utilization: number;
}

/** One WebSocket frame from the simulator. */
export interface StateFrame {
  state: NetworkState;
  metrics: StepMetrics;
}

export type ConnectionStatus = 'connecting' | 'live' | 'stale' | 'disconnected';
