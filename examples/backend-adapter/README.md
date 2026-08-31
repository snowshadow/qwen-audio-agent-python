# Backend Adapter SDK example

`in-memory-backend.mjs` is a minimal non-ACP `BackendPort` implementation. It
shows lifecycle, Task submission, status, cancellation, events, bounded public
results, and idempotent cleanup without exposing a private task graph.

Run the same public conformance suite used by built-in adapters:

```bash
node --test server/test/backend-adapter-sdk.test.mjs
```

For an adapter self-check, wrap it with `createBackendAgentHost` and run
`verifyBackendAdapterConformance`. v2 no longer provides an embedded JavaScript
Gateway. Product-specific credentials, transport clients, and retries stay
inside the adapter.
