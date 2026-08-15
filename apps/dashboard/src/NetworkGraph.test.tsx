import { describe, expect, it } from 'vitest';
import { failedNodeIds, layoutNodes, utilizationColor } from './graph';

describe('utilizationColor', () => {
  it('escalates through bands as utilization rises', () => {
    const colors = [0.1, 0.5, 0.75, 0.95].map(utilizationColor);
    expect(new Set(colors).size).toBe(4);
  });

  it('flags saturation distinctly from heavy load', () => {
    expect(utilizationColor(0.95)).not.toBe(utilizationColor(0.75));
  });
});

describe('layoutNodes', () => {
  it('places every node', () => {
    const positions = layoutNodes(['n0', 'n1', 'n2'], 400, 400);
    expect(positions.size).toBe(3);
  });

  it('is deterministic, so the topology does not reshuffle between frames', () => {
    const first = layoutNodes(['n0', 'n1', 'n2'], 400, 400);
    const second = layoutNodes(['n0', 'n1', 'n2'], 400, 400);
    expect([...first.entries()]).toEqual([...second.entries()]);
  });

  it('keeps nodes inside the viewport', () => {
    const positions = layoutNodes(['n0', 'n1', 'n2', 'n3'], 400, 400);
    for (const { x, y } of positions.values()) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(400);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(400);
    }
  });
});

describe('failed node detection', () => {
  const link = (id: string, source: string, target: string, effective: number) => ({
    link_id: id,
    source,
    target,
    capacity_mbps: 100,
    effective_capacity_mbps: effective,
    allocated_mbps: 0,
    drained_mbps: 0,
    queue_mb: 0,
    utilization: 0,
    latency_ms: 1,
  });

  it('does not mark healthy endpoints of a single failed link as failed', () => {
    // A one-link failure used to paint both of its nodes red, overstating the incident.
    const links = [
      link('n0->n1', 'n0', 'n1', 0),
      link('n0->n2', 'n0', 'n2', 100),
      link('n1->n2', 'n1', 'n2', 100),
    ];
    const failed = failedNodeIds(links);
    expect(failed.size).toBe(0);
  });

  it('marks a node failed when every link touching it is down', () => {
    const links = [link('n0->n1', 'n0', 'n1', 0), link('n2->n1', 'n2', 'n1', 0)];
    expect(failedNodeIds(links).has('n1')).toBe(true);
  });
});
