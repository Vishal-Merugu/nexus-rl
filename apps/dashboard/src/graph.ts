/**
 * Pure geometry and colour helpers for the network view.
 *
 * Split out of `NetworkGraph.tsx` so that file exports a component and nothing else: React Fast Refresh
 * silently degrades to a full reload for any module that mixes components with other exports, which
 * makes the live view flicker and lose its WebSocket during development.
 *
 * Everything here is a pure function of its arguments — which is what lets the tests cover the layout
 * and colour banding without rendering anything.
 */

import type { LinkState } from './types';

export interface Point {
  x: number;
  y: number;
}

/** Utilization → colour. Steps, not a gradient: a reader should be able to name the band. */
export function utilizationColor(utilization: number): string {
  if (utilization >= 0.9) return '#b91c1c';
  if (utilization >= 0.7) return '#d97706';
  if (utilization >= 0.4) return '#0891b2';
  return '#15803d';
}

/** Stroke width in px, so allocated rate reads at a glance without a legend. */
export function linkWidth(link: LinkState): number {
  const share = link.capacity_mbps > 0 ? link.allocated_mbps / link.capacity_mbps : 0;
  return 1.5 + share * 6;
}

/**
 * Nodes whose every incident link is down — a genuine node failure.
 *
 * Marking both endpoints of any single failed link painted two nodes red for what is really one link
 * failure, overstating the incident during exactly the recovery demo this view exists for.
 */
export function failedNodeIds(links: LinkState[]): Set<string> {
  const total = new Map<string, number>();
  const down = new Map<string, number>();
  for (const link of links) {
    for (const node of [link.source, link.target]) {
      total.set(node, (total.get(node) ?? 0) + 1);
      if (link.effective_capacity_mbps === 0) down.set(node, (down.get(node) ?? 0) + 1);
    }
  }
  return new Set(
    [...total.entries()]
      .filter(([node, count]) => (down.get(node) ?? 0) === count)
      .map(([node]) => node),
  );
}

/** Nodes on a circle, ordered by id. Deterministic — the same topology always draws the same way. */
export function layoutNodes(nodeIds: string[], width: number, height: number): Map<string, Point> {
  const radius = Math.min(width, height) / 2 - 48;
  const centerX = width / 2;
  const centerY = height / 2;
  const positions = new Map<string, Point>();

  nodeIds.forEach((id, index) => {
    const angle = (2 * Math.PI * index) / nodeIds.length - Math.PI / 2;
    positions.set(id, {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    });
  });
  return positions;
}
