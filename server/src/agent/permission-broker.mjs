import { randomUUID } from 'node:crypto'
import { redactLogValue } from '../../../shared/logger.mjs'
import { AgentError } from './backend-adapter.mjs'
import {
  ACP_SESSION_TOOL_NAMES,
  ACP_SESSION_TOOL_SERVER,
} from './acp-session-tools.mjs'
import {
  AuthorizationStatus,
  normalizeAuthorization,
  resolveAuthorization,
} from '../core/work-authorization.mjs'
import { BackendEventType, backendEvent } from '../core/backend-events.mjs'

function clean(value) {
  return String(value || '').trim()
}

function bounded(value, max = 300) {
  return clean(value).replace(/\s+/g, ' ').slice(0, max)
}

function safeField(value, key, max) {
  if (value === null || value === undefined) return ''
  const redacted = redactLogValue({ [key]: value })?.[key]
  if (Array.isArray(redacted)) return bounded(redacted.join(' '), max)
  if (typeof redacted === 'object') return ''
  return bounded(redacted, max)
}

function permissionOperation(toolCall = {}) {
  const rawInput = toolCall.rawInput && typeof toolCall.rawInput === 'object'
    ? toolCall.rawInput
    : {}
  const title = safeField(
    toolCall.title || toolCall.name || '后台操作',
    'title',
    160,
  )
  const description = safeField(
    rawInput.description || rawInput.query,
    'description',
    600,
  )
  const command = safeField(rawInput.command, 'command', 1200)
  const path = safeField(
    rawInput.path || rawInput.filePath || rawInput.file_path,
    'path',
    600,
  )
  const locations = (Array.isArray(toolCall.locations) ? toolCall.locations : [])
    .map(location => {
      const locationPath = safeField(location?.path, 'path', 600)
      if (!locationPath) return null
      return {
        path: locationPath,
        ...(Number.isInteger(location?.line) && location.line > 0
          ? { line: location.line }
          : {}),
      }
    })
    .filter(Boolean)
    .slice(0, 16)
  return {
    title,
    kind: safeField(toolCall.kind || toolCall.name, 'kind', 80) || 'unknown',
    ...(description ? { description } : {}),
    ...(command ? { command } : {}),
    ...(path ? { path } : {}),
    ...(locations.length ? { locations } : {}),
  }
}

function permissionSummary(operation) {
  const detail = operation.description
    || operation.command
    || operation.path
    || operation.locations?.[0]?.path
    || ''
  return [operation.title, detail]
    .filter((value, index, values) => value && values.indexOf(value) === index)
    .join('：') || '后台操作'
}

function deferred() {
  let resolvePromise
  const promise = new Promise(resolve => { resolvePromise = resolve })
  return { promise, resolve: resolvePromise }
}

function optionFor(params, decision, { allowPersistent = false } = {}) {
  const options = Array.isArray(params?.options) ? params.options : []
  // A public approval resolves this request only, so it must map to
  // `allow_once`. Full-permission/internal flows may explicitly opt into the
  // persistent fallback. Rejection still prefers the narrow one-shot option.
  const kinds = decision === 'always'
    ? allowPersistent ? ['allow_once', 'allow_always'] : ['allow_once']
    : ['reject_once', 'reject_always']
  return kinds
    .map(kind => options.find(candidate => (
      candidate?.kind === kind && clean(candidate.optionId)
    )))
    .find(Boolean) || null
}

function matchingInternalSessionToolName(candidate) {
  for (const toolName of ACP_SESSION_TOOL_NAMES) {
    const accepted = new Set([
      toolName,
      `${ACP_SESSION_TOOL_SERVER}__${toolName}`,
      `mcp__${ACP_SESSION_TOOL_SERVER}__${toolName}`,
      `${toolName} (${ACP_SESSION_TOOL_SERVER})`,
    ])
    if (accepted.has(candidate)) return toolName
  }
  return ''
}

function internalSessionToolName(toolCall = {}) {
  const name = clean(toolCall.name)
  // ACP defines `name` as the programmatic identity and `title` as display
  // text. A present non-internal name must never be overridden by a title.
  if (name) return matchingInternalSessionToolName(name)
  return matchingInternalSessionToolName(clean(toolCall.title))
}

export class PermissionBroker {
  constructor({
    protocol,
    permissionMode,
    resolvedLimit = 200,
    isTrustedInternalToolRequest,
  }) {
    this.protocol = protocol
    this.permissionMode = permissionMode
    this.resolvedLimit = resolvedLimit
    this.pending = new Map()
    this.queued = new Map()
    this.scopeQueues = new Map()
    this.activeByScope = new Map()
    this.scheduledScopes = new Set()
    this.resolved = new Map()
    this.isTrustedInternalToolRequest = isTrustedInternalToolRequest
  }

  scopeKey(session = {}) {
    const taskId = clean(session.coordinationRunId)
    if (taskId) return `task:${taskId}`
    const permissionScopeId = clean(session.permissionScopeId)
    if (permissionScopeId) return `prompt:${permissionScopeId}`
    const sessionId = clean(session.sessionId)
    if (sessionId) return `session:${sessionId}`
    return `owner:${clean(session.ownerId) || 'anonymous'}`
  }

  activateNext(scopeKey) {
    this.scheduledScopes.delete(scopeKey)
    if (this.activeByScope.has(scopeKey)) return
    const queue = this.scopeQueues.get(scopeKey) || []
    let record = null
    while (queue.length && !record) {
      const candidate = queue.shift()
      if (!candidate.settled) record = candidate
    }
    if (!queue.length) this.scopeQueues.delete(scopeKey)
    if (!record) return
    this.queued.delete(record.id)
    record.exposed = true
    this.pending.set(record.id, record)
    this.activeByScope.set(scopeKey, record.id)
    record.onEvent?.(backendEvent(
      BackendEventType.AUTHORIZATION_REQUESTED,
      { permission: record.permission },
    ))
  }

  scheduleNext(scopeKey) {
    if (this.scheduledScopes.has(scopeKey)) return
    this.scheduledScopes.add(scopeKey)
    setImmediate(() => this.activateNext(scopeKey))
  }

  enqueue(record) {
    const queue = this.scopeQueues.get(record.scopeKey) || []
    queue.push(record)
    this.scopeQueues.set(record.scopeKey, queue)
    this.queued.set(record.id, record)
    if (!this.activeByScope.has(record.scopeKey)
      && !this.scheduledScopes.has(record.scopeKey)) {
      this.activateNext(record.scopeKey)
    }
  }

  removeQueued(record) {
    this.queued.delete(record.id)
    const queue = this.scopeQueues.get(record.scopeKey)
    if (!queue) return
    const index = queue.indexOf(record)
    if (index >= 0) queue.splice(index, 1)
    if (!queue.length) this.scopeQueues.delete(record.scopeKey)
  }

  rememberResolved(permission, ownerId) {
    this.resolved.set(permission.id, { ownerId, permission })
    while (this.resolved.size > this.resolvedLimit) {
      this.resolved.delete(this.resolved.keys().next().value)
    }
  }

  finish(record, status, outcome, { activateNext = true } = {}) {
    if (!record || record.settled) return null
    record.settled = true
    record.signal?.removeEventListener?.('abort', record.onAbort)
    const wasActive = this.pending.delete(record.id)
    if (wasActive) this.activeByScope.delete(record.scopeKey)
    else this.removeQueued(record)
    record.pending.resolve(outcome)
    const permission = resolveAuthorization(record, status)
    if (record.exposed && permission) {
      record.onEvent?.(backendEvent(
        BackendEventType.AUTHORIZATION_RESOLVED,
        { permission },
      ))
      this.rememberResolved(permission, record.ownerId)
    }
    if (wasActive && activateNext) this.scheduleNext(record.scopeKey)
    return permission
  }

  async request(params, { signal, session } = {}) {
    const name = clean(params?.toolCall?.name || params?.toolCall?.title)
    const internalToolName = internalSessionToolName(params?.toolCall)
    // A backend can choose tool names and titles. They identify the requested
    // capability, but only the adapter can attest that this exact Session is
    // currently bound to the Gateway-owned MCP registration and owner/run.
    let internal = false
    if (internalToolName && this.isTrustedInternalToolRequest) {
      try {
        internal = this.isTrustedInternalToolRequest({
          params,
          session,
          toolName: internalToolName,
        }) === true
      } catch {
        internal = false
      }
    }
    if (this.permissionMode === 'full' || internal) {
      const option = optionFor(params, 'always', { allowPersistent: true })
      return option
        ? { outcome: { outcome: 'selected', optionId: option.optionId } }
        : { outcome: { outcome: 'cancelled' } }
    }
    if (!session || typeof session !== 'object') {
      return { outcome: { outcome: 'cancelled' } }
    }
    const id = `auth_${randomUUID().replaceAll('-', '')}`
    const pending = deferred()
    const operation = permissionOperation(params?.toolCall)
    const permission = normalizeAuthorization({
      id,
      taskId: session?.coordinationRunId || null,
      status: AuthorizationStatus.PENDING,
      category: operation.kind || bounded(name, 80) || 'unknown',
      summary: permissionSummary(operation),
      patterns: [],
      approvalScope: 'once',
      operation,
    })
    const record = {
      ...permission,
      permission,
      ownerId: clean(session?.ownerId),
      sessionId: clean(session?.sessionId),
      permissionScopeId: clean(session?.permissionScopeId),
      scopeKey: this.scopeKey(session),
      params,
      pending,
      onEvent: session?.onEvent,
      signal,
      exposed: false,
      settled: false,
    }
    record.onAbort = () => this.cancel(record)
    if (signal?.aborted) {
      record.settled = true
      record.pending.resolve({ outcome: { outcome: 'cancelled' } })
      return pending.promise
    }
    signal?.addEventListener?.('abort', record.onAbort, { once: true })
    this.enqueue(record)
    return pending.promise
  }

  cancel(record, { activateNext = true } = {}) {
    return Boolean(this.finish(
      record,
      AuthorizationStatus.CANCELLED,
      { outcome: { outcome: 'cancelled' } },
      { activateNext },
    ))
  }

  respond(id, decision, { ownerId } = {}) {
    const key = String(id)
    const record = this.pending.get(key)
    if (!record) {
      const resolved = this.resolved.get(key)
      if (resolved?.ownerId === clean(ownerId)) return resolved.permission
    }
    if (!record || record.ownerId !== clean(ownerId)) {
      throw new AgentError('权限请求不存在、已经失效或不属于当前用户', {
        protocol: this.protocol,
      })
    }
    const approved = decision === 'always'
    const option = optionFor(
      record.params,
      approved ? 'always' : 'reject',
    )
    const outcome = option
      ? { outcome: { outcome: 'selected', optionId: option.optionId } }
      : { outcome: { outcome: 'cancelled' } }
    const permission = this.finish(
      record,
      option
        ? approved ? AuthorizationStatus.APPROVED : AuthorizationStatus.DENIED
        : AuthorizationStatus.CANCELLED,
      outcome,
    )
    return permission
  }

  cancelScope(permissionScopeId) {
    const scope = clean(permissionScopeId)
    if (!scope) return
    const records = [...this.pending.values(), ...this.queued.values()]
      .filter(record => record.permissionScopeId === scope)
    const affectedScopes = new Set()
    for (const record of records) {
      affectedScopes.add(record.scopeKey)
      this.cancel(record, { activateNext: false })
    }
    for (const scopeKey of affectedScopes) {
      if (!this.activeByScope.has(scopeKey)) this.scheduleNext(scopeKey)
    }
  }

  cancelAll() {
    for (const record of [...this.pending.values(), ...this.queued.values()]) {
      this.cancel(record, { activateNext: false })
    }
    this.scopeQueues.clear()
    this.activeByScope.clear()
    this.scheduledScopes.clear()
  }
}
