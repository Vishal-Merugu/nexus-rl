/**
 * Dashboard shell.
 *
 * A pure consumer: WebSocket in, pixels out. No QoS maths the backend should own, no writes to the
 * simulator. Operator instructions go through the agent layer, which is the only thing that may change
 * anything — and even then only constraints (ADR-0002).
 *
 * Every panel renders `frame`, so they all show the same timestep. Showing two timesteps at once is worse
 * than lagging by one.
 */

import { NetworkGraph } from './NetworkGraph';
import { useNetworkStream } from './useNetworkStream';
import type { ConnectionStatus } from './types';

const SIMULATOR_WS = import.meta.env.VITE_SIMULATOR_WS ?? 'ws://localhost:8000/ws';

const STATUS_LABEL: Record<ConnectionStatus, string> = {
  connecting: 'Connecting…',
  live: 'Live',
  stale: 'Stale — no data for 5s',
  disconnected: 'Disconnected — retrying',
};

export function App() {
  const { frame, status, history } = useNetworkStream(SIMULATOR_WS);

  return (
    <main className="dashboard">
      <header>
        <h1>NEXUS-RL</h1>
        <span className={`status status--${status}`}>{STATUS_LABEL[status]}</span>
        {frame && <span className="timestep">t = {frame.state.timestep}</span>}
      </header>

      {!frame ? (
        <p className="empty">
          Waiting for the simulator. Start it with{' '}
          <code>docker compose -f .docker/docker-compose.yml up -d</code>.
        </p>
      ) : (
        <div className="panels">
          <section aria-label="Network topology">
            <NetworkGraph state={frame.state} />
            {frame.state.active_incident_ids.length > 0 && (
              <p className="incident">
                Incident active: {frame.state.active_incident_ids.join(', ')}
              </p>
            )}
          </section>

          <section aria-label="Quality of service metrics">
            <dl className="metrics">
              <Metric label="p95 latency" value={`${frame.metrics.p95_latency_ms.toFixed(1)} ms`} />
              <Metric
                label="Throughput"
                value={`${frame.metrics.total_throughput_mbps.toFixed(0)} Mb/s`}
              />
              <Metric label="Fairness" value={frame.metrics.fairness_index.toFixed(3)} />
              <Metric
                label="Dropped"
                value={`${frame.metrics.total_dropped_mbps.toFixed(1)} Mb/s`}
              />
            </dl>
            <p className="note">
              Fairness is shown beside throughput deliberately — they trade off, and either alone misleads.
            </p>
          </section>

          <section aria-label="Flows">
            <table className="flows">
              <thead>
                <tr>
                  <th>Flow</th>
                  <th>Class</th>
                  <th>Demand</th>
                  <th>Allocated</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody>
                {frame.state.flows.map((flow) => (
                  <tr key={flow.flow_id}>
                    <td>{flow.flow_id}</td>
                    <td>{flow.traffic_class}</td>
                    <td>{flow.demand_mbps.toFixed(0)}</td>
                    <td>{flow.allocated_mbps.toFixed(0)}</td>
                    <td>{flow.latency_ms.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section aria-label="History">
            <p className="note">{history.length} frames buffered for time-series panels.</p>
          </section>
        </div>
      )}
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
