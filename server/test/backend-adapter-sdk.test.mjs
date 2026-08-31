import assert from 'node:assert/strict'
import test from 'node:test'
import {
  BACKEND_ADAPTER_SDK_VERSION,
  BackendEventType,
  BackendWorkRuntime,
  createBackendAgentHost,
  defineBackendAdapter,
  verifyBackendAdapterConformance,
} from '../src/backend/backend-adapter-sdk.mjs'
import {
  createConformanceFixture,
  createInMemoryBackend,
} from '../../examples/backend-adapter/in-memory-backend.mjs'

test('exports one stable Backend Adapter SDK entry point', () => {
  assert.equal(BACKEND_ADAPTER_SDK_VERSION, '2.0.0')
  assert.equal(BackendEventType.MESSAGE, 'backend.message')
  const backend = createInMemoryBackend()
  assert.equal(defineBackendAdapter(backend), backend)
  const host = createBackendAgentHost(backend)
  assert.equal(host.enabled, true)
  assert.equal(host.protocol, 'example-memory')
  assert.equal(host.status().status, 'stopped')
})

test('the non-ACP SDK example passes the public conformance suite', async () => {
  await verifyBackendAdapterConformance({
    createFixture: createConformanceFixture,
  })
})

test('a custom BackendPort runs through BackendWorkRuntime without ACP', async () => {
  const backend = createInMemoryBackend()
  const runtime = new BackendWorkRuntime({ backend })
  const outcome = await runtime.run({
    originalRequest: 'Check the custom adapter',
    objective: 'Check the custom adapter',
  }, {
    taskId: 'task_1',
    ownerId: 'sdk-owner',
  })
  assert.match(outcome.content, /Check the custom adapter/)
})
