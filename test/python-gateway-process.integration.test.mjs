import assert from 'node:assert/strict'
import { fork } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { createGatewayProcess } from '../shared/gateway-process.mjs'

async function waitUntil(predicate, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) return false
    await new Promise(resolve => setTimeout(resolve, 20))
  }
  return true
}

test('the unchanged Node host starts and stops the Python Gateway', async t => {
  const configDir = mkdtempSync(join(tmpdir(), 'qwaudio-python-host-'))
  let childOutput = ''
  const gateway = createGatewayProcess({
    configDir,
    backend: 'none',
    wakeWord: false,
    preferredPort: 0,
    env: {
      ...process.env,
      QWEN_AUDIO_ALLOW_UNCONFIGURED: '1',
      QWEN_AUDIO_LOG_LEVEL: 'warning',
    },
    forkImpl: (entry, args, options) => {
      const child = fork(entry, args, {
        ...options,
        stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
      })
      child.stdout.on('data', chunk => { childOutput += chunk })
      child.stderr.on('data', chunk => { childOutput += chunk })
      return child
    },
    startupTimeoutMs: 15_000,
    stopTimeoutMs: 5_000,
  })

  let origin
  try {
    origin = await gateway.start()
  } catch (error) {
    if (childOutput.includes('Operation not permitted')) {
      t.skip('sandbox does not allow binding a loopback socket')
      return
    }
    throw error
  }
  try {
    const health = await fetch(`${origin}/api/health`).then(response => response.json())
    const lease = JSON.parse(readFileSync(join(configDir, 'gateway.lock'), 'utf8'))
    assert.equal(health.protocolVersion, '4.0.0')
    assert.equal(health.gatewayInstanceId, lease.instanceId)
    assert.equal(lease.origin, origin)
  } finally {
    await gateway.stop()
  }
  assert.equal(
    await waitUntil(() => !existsSync(join(configDir, 'gateway.lock'))),
    true,
    childOutput,
  )
})
