/**
 * The network graph — the centrepiece of the demo.
 *
 * Node positions are computed **once** from the node set and then frozen. A force layout that reshuffles
 * every tick is unreadable and makes a screen recording look like noise; the interesting motion is in the
 * links, not the topology.
 *
 * Utilization is encoded by colour *and* a numeric label. Colour alone fails for colour-blind viewers and
 * washes out in a compressed screen recording — both of which matter for this project's actual audience.
 *
 * Layout and colour helpers live in `./graph` so this module exports only a component, which is what
 * React Fast Refresh requires to hot-update instead of reloading the page.
 */

import { useMemo } from 'react';
import { failedNodeIds, layoutNodes, linkWidth, utilizationColor } from './graph';
import type { NetworkState } from './types';

interface Props {
  state: NetworkState;
  width?: number;
  height?: number;
}

export function NetworkGraph({ state, width = 560, height = 460 }: Props) {
  const nodeIds = useMemo(() => {
    const ids = new Set<string>();
    for (const link of state.links) {
      ids.add(link.source);
      ids.add(link.target);
    }
    return [...ids].sort();
  }, [state.links]);

  // Positions depend only on the node set, so they survive every state update.
  const positions = useMemo(() => layoutNodes(nodeIds, width, height), [nodeIds, width, height]);

  const failedNodes = useMemo(() => failedNodeIds(state.links), [state.links]);

  return (
    <svg width={width} height={height} role="img" aria-label="Network topology and link utilization">
      {state.links.map((link) => {
        const from = positions.get(link.source);
        const to = positions.get(link.target);
        if (!from || !to) return null;

        const degraded = link.effective_capacity_mbps < link.capacity_mbps;
        return (
          <g key={link.link_id}>
            <line
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={degraded ? '#6b7280' : utilizationColor(link.utilization)}
              strokeWidth={linkWidth(link)}
              strokeDasharray={degraded ? '4 3' : undefined}
              strokeLinecap="round"
              opacity={0.85}
            />
            <text
              x={(from.x + to.x) / 2}
              y={(from.y + to.y) / 2 - 4}
              fontSize={10}
              fill="#374151"
              textAnchor="middle"
            >
              {Math.round(link.utilization * 100)}%
            </text>
          </g>
        );
      })}

      {nodeIds.map((id) => {
        const point = positions.get(id);
        if (!point) return null;
        const failed = failedNodes.has(id);
        return (
          <g key={id}>
            <circle
              cx={point.x}
              cy={point.y}
              r={18}
              fill={failed ? '#fee2e2' : '#ffffff'}
              stroke={failed ? '#b91c1c' : '#1f2937'}
              strokeWidth={2}
            />
            <text x={point.x} y={point.y + 4} fontSize={12} textAnchor="middle" fill="#111827">
              {id}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
