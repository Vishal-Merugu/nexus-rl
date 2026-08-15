/**
 * The single WebSocket subscription.
 *
 * One connection, one reducer, one source of truth. Multiple sockets or per-component subscriptions produce
 * panels that disagree about the current timestep — which looks like a backend bug and is not one.
 *
 * Frames are coalesced to animation frames rather than set on state per message. The simulator can step far
 * faster than a browser can paint, and a `setState` per frame stalls the tab during exactly the busy periods
 * that are most interesting to watch.
 */

import { useEffect, useRef, useState } from 'react';
import type { ConnectionStatus, StateFrame } from './types';

const RECONNECT_BASE_MS = 500;
const RECONNECT_MAX_MS = 8_000;
/** No frame for this long means the feed is stale, even though the socket is technically open. */
const STALE_AFTER_MS = 5_000;

export interface NetworkStream {
  frame: StateFrame | null;
  status: ConnectionStatus;
  /** Recent frames, oldest first, for time-series panels. */
  history: StateFrame[];
}

export function useNetworkStream(url: string, historyLength = 120): NetworkStream {
  const [frame, setFrame] = useState<StateFrame | null>(null);
  const [history, setHistory] = useState<StateFrame[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>('connecting');

  const pending = useRef<StateFrame | null>(null);
  const rafHandle = useRef<number | null>(null);
  // Seeded inside the effect, not here: `Date.now()` during render is an impure call, and under
  // StrictMode's double render the two passes disagree. It also started the staleness clock at render
  // time rather than at connect time, so a slow initial connection could report "stale" before the
  // socket had ever opened.
  const lastFrameAt = useRef<number>(0);

  useEffect(() => {
    lastFrameAt.current = Date.now();
    let socket: WebSocket | null = null;
    let reconnectDelay = RECONNECT_BASE_MS;
    let reconnectTimer: number | undefined;
    let closed = false;

    const flush = () => {
      rafHandle.current = null;
      const next = pending.current;
      if (!next) return;
      pending.current = null;
      setFrame(next);
      setHistory((previous) => [...previous, next].slice(-historyLength));
    };

    const connect = () => {
      socket = new WebSocket(url);

      socket.onopen = () => {
        setStatus('live');
        reconnectDelay = RECONNECT_BASE_MS;
      };

      socket.onmessage = (event: MessageEvent<unknown>) => {
        lastFrameAt.current = Date.now();
        setStatus('live');
        // `event.data` is `any` on the DOM type. A binary frame would make `JSON.parse` throw inside
        // the handler and silently kill the subscription, so non-text frames are ignored explicitly.
        if (typeof event.data !== 'string') return;
        pending.current = JSON.parse(event.data) as StateFrame;
        // Coalesce: only the newest frame survives until the next paint.
        if (rafHandle.current === null) {
          rafHandle.current = requestAnimationFrame(flush);
        }
      };

      socket.onclose = () => {
        if (closed) return;
        setStatus('disconnected');
        // Backoff, so a restarting simulator is not hammered by every open tab.
        reconnectTimer = window.setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
      };

      socket.onerror = () => socket?.close();
    };

    connect();

    // A frozen chart that looks live is a lie, and during a demo an embarrassing one.
    const staleCheck = window.setInterval(() => {
      if (Date.now() - lastFrameAt.current > STALE_AFTER_MS) {
        setStatus((current) => (current === 'disconnected' ? current : 'stale'));
      }
    }, 1_000);

    return () => {
      closed = true;
      window.clearInterval(staleCheck);
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (rafHandle.current !== null) cancelAnimationFrame(rafHandle.current);
      socket?.close();
    };
  }, [url, historyLength]);

  return { frame, status, history };
}
