/// <reference types="vite/client" />

/**
 * Declared so `import.meta.env.VITE_SIMULATOR_WS` is `string | undefined` rather than `any`.
 *
 * vite/client types the env bag with an `any` index signature, so every read of a custom variable
 * silently produced an `any` that then flowed into `useNetworkStream(url)` unchecked — a renamed or
 * missing variable would have reached the WebSocket constructor as `undefined` instead of failing here.
 */
interface ImportMetaEnv {
  /** WebSocket URL of the simulator's state stream. Falls back to localhost when unset. */
  readonly VITE_SIMULATOR_WS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
