// Public compatibility entry retained for existing Node and Electron hosts.
// Gateway behavior now lives in Python; this file only launches the bridge.
await import('./python-gateway-entry.mjs')
