import { createInterface } from 'node:readline'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createLogger } from '../../shared/logger.mjs'
import { startManagedBackend } from './process/managed-backend.mjs'

const sourceRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const root = process.env.QWEN_AUDIO_AGENT_RUNTIME_ROOT || sourceRoot
const logger = createLogger({ component: 'python-backend-bridge', fileName: 'gateway.log' })

let backendRuntime = null
let agent = null
let unsubscribe = null
let closing = null

function write(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`)
}

function publicError(error) {
  return {
    message: String(error?.message || error || 'Backend bridge failed'),
    code: typeof error?.code === 'string' ? error.code : 'BACKEND_BRIDGE_ERROR',
  }
}

async function ensureStarted(context = {}) {
  if (agent) return agent
  backendRuntime = await startManagedBackend({ root, logger })
  const module = await import('./agent/agent-client.mjs')
  agent = module.agent
  unsubscribe = agent.subscribe(event => write({ event }))
  await agent.start(context)
  return agent
}

async function close() {
  if (closing) return closing
  closing = Promise.allSettled([
    Promise.resolve().then(() => unsubscribe?.()),
    Promise.resolve().then(() => agent?.close()),
    Promise.resolve().then(() => backendRuntime?.stop('SIGTERM')),
  ]).finally(() => {
    unsubscribe = null
    agent = null
    backendRuntime?.close?.()
    backendRuntime = null
  })
  return closing
}

async function dispatch(method, params = {}) {
  if (method === 'describe') {
    const current = await ensureStarted(params.context)
    return current.describe()
  }
  if (method === 'start') {
    const current = await ensureStarted(params.context)
    return current.health()
  }
  if (method === 'close') {
    await close()
    return { ok: true }
  }
  const current = await ensureStarted(params.context)
  if (method === 'health') return current.health()
  if (method === 'submit') return current.submit(params.task, params.context)
  if (method === 'status') return current.status(params.taskId, params.context)
  if (method === 'cancel') return current.cancel(params.taskId, params.context)
  if (method === 'respondAuthorization') {
    return current.respondAuthorization(
      params.taskId,
      params.authorizationId,
      params.decision,
      params.context,
    )
  }
  if (method === 'canRecoverDelegatedWork') {
    return current.validateRecoverableDelegatedWork?.(params.task)
      ?? current.canRecoverDelegatedWork(params.task)
  }
  if (method === 'recoverDelegatedWork') {
    return current.recoverDelegatedWork(params.task, params.context)
  }
  throw Object.assign(new Error(`Unknown bridge method: ${method}`), {
    code: 'BACKEND_BRIDGE_METHOD_NOT_FOUND',
  })
}

createInterface({ input: process.stdin }).on('line', line => {
  let request
  try {
    request = JSON.parse(line)
  } catch {
    return
  }
  if (!request?.id || typeof request.method !== 'string') return
  Promise.resolve(dispatch(request.method, request.params)).then(
    result => write({ id: request.id, result }),
    error => write({ id: request.id, error: publicError(error) }),
  )
})

async function stop(signal) {
  await close()
  process.exit(signal === 'SIGINT' ? 130 : 0)
}

process.once('SIGINT', () => { stop('SIGINT') })
process.once('SIGTERM', () => { stop('SIGTERM') })
process.stdin.once('end', () => { stop('SIGTERM') })
