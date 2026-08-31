import { randomUUID } from 'node:crypto'
import { AgentError } from './backend-adapter.mjs'
import {
  COORDINATOR_MCP_INSTRUCTIONS_MAX_BYTES,
  COORDINATOR_STABLE_INSTRUCTIONS,
} from './acp-coordinator-instructions.mjs'
import { BackendEventType, backendEvent } from '../core/backend-events.mjs'
import {
  acpBackendProfile,
  endpointAvailable,
} from './acp-backend-profile.mjs'
import {
  activityFromUpdate,
  coordinatorKey,
  legacyProjectSessionKey,
  messageFromUpdate,
  nativeToolOutput,
  projectSessionKey,
  sessionOwnerId,
  sessionSummary,
} from './acp-backend-session-utils.mjs'
import { createAcpClient } from './acp-client-factory.mjs'
import { AcpSessionRegistry } from './acp-session-registry.mjs'
import { AcpSessionToolServer } from './acp-session-tools.mjs'
import {
  builtinMcpServers,
  createBuiltinMcpLifecycle,
} from './builtin-mcp.mjs'
import { BackendRuntimeState } from './backend-runtime-state.mjs'
import { KeyedSerialExecutor } from './keyed-serial-executor.mjs'
import { PermissionBroker } from './permission-broker.mjs'
import {
  appendPromptBlocks,
  artifactsFromAcpContentBlocks,
  nonTextPromptBlocks,
  promptWithInputParts,
  transformPromptText,
} from './acp-content.mjs'
import { assertMcpServerCapabilities } from './acp-capabilities.mjs'
import { buildAcpCoordinatorInstruction } from './acp-coordinator-contract.mjs'

const MAX_SESSION_RESULTS = 100
const MAX_DELEGATION_RESULT_CHARS = 12_000
const MAX_DELEGATION_RECENT_UPDATES = 5
const DEFAULT_RECOVERY_PROBE_TIMEOUT_MS = 5_000
// Persistent coordinator Sessions are valid only for the contract that
// created them. Project Sessions are user work and remain independent.
const COORDINATOR_CONTRACT_VERSION = 6

export { acpBackendProfile } from './acp-backend-profile.mjs'

function clean(value) {
  return String(value || '').trim()
}

function assertCompletedAcpTurn(result, {
  signal,
  label = 'ACP Session',
  protocol = 'acp',
} = {}) {
  const stopReason = clean(result?.response?.stopReason)
  if (stopReason === 'end_turn') return result
  if (stopReason === 'cancelled') {
    throw signal?.reason || new AgentError(`${label} 已取消`, {
      status: 499,
      protocol,
    })
  }
  const reason = {
    max_tokens: '达到最大输出长度，未能完成当前回合',
    max_turn_requests: '达到最大 Agent 请求次数，未能完成当前回合',
    refusal: '拒绝继续当前回合',
  }[stopReason] || `返回了未知的终止原因 ${stopReason || '(missing)'}`
  throw new AgentError(`${label} ${reason}`, {
    status: 502,
    protocol,
    body: clean(result?.content),
  })
}

function explicitModel(value) {
  const model = clean(value)
  return model.toLowerCase() === 'auto' ? '' : model
}

function modelKey(value) {
  return clean(value).toLowerCase()
}

function optionChoices(entries = []) {
  return entries.flatMap(entry => {
    if (Array.isArray(entry?.options)) return optionChoices(entry.options)
    const value = clean(entry?.value)
    if (!value) return []
    return [{
      value,
      names: [entry?.name, entry?.label]
        .map(clean)
        .filter(Boolean),
    }]
  })
}

function matchingOptionValue(entries, desired) {
  const desiredKey = modelKey(desired)
  const choice = optionChoices(entries).find(item => (
    modelKey(item.value) === desiredKey
    || item.names.some(name => modelKey(name) === desiredKey)
  ))
  return choice?.value || ''
}

function bounded(value, max = 300) {
  return clean(value).replace(/\s+/g, ' ').slice(0, max)
}

function optionValues(entries = []) {
  return optionChoices(entries).map(entry => entry.value)
}

function modelConfigOption(options = []) {
  return options.find(option => clean(option?.category).toLowerCase() === 'model')
    || options.find(option => (
      ['model', 'models'].includes(clean(option?.id).toLowerCase())
    ))
    || null
}

function legacyModelState(response) {
  const models = response?.models
  if (!models || !Array.isArray(models.availableModels)) return null
  return {
    currentValue: clean(models.currentModelId),
    choices: models.availableModels
      .map(item => ({
        value: clean(item?.modelId),
        names: [item?.name].map(clean).filter(Boolean),
      }))
      .filter(item => item.value),
  }
}

function deferred() {
  let resolvePromise
  let rejectPromise
  const promise = new Promise((resolve, reject) => {
    resolvePromise = resolve
    rejectPromise = reject
  })
  return { promise, resolve: resolvePromise, reject: rejectPromise }
}

function waitForRetry(delayMs, signal) {
  if (signal?.aborted) {
    return Promise.reject(signal.reason || new Error('任务已取消'))
  }
  return new Promise((resolvePromise, rejectPromise) => {
    const finish = () => {
      signal?.removeEventListener('abort', abort)
      resolvePromise()
    }
    const timer = setTimeout(finish, delayMs)
    const abort = () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      rejectPromise(signal.reason || new Error('任务已取消'))
    }
    signal?.addEventListener('abort', abort, { once: true })
  })
}

export class AcpBackendAdapter {
  constructor({
    protocol = 'acp',
    root = process.cwd(),
    ownership = 'owned',
    permissionMode = 'native',
    model = '',
    timeoutMs = 300_000,
    directory = process.cwd(),
    cliPath = '',
    configDirectory = '',
    claudeExecutable = '',
    baseUrl = '',
    token = '',
    tokenFile = '',
    coordinatorAgent = '',
    profile,
    sessionStatePath = null,
    client,
    clientFactory = createAcpClient,
    backendAvailable = endpointAvailable,
    readinessPollMs = 250,
    readinessTimeoutMs = Math.min(timeoutMs, 60_000),
    recoveryProbeTimeoutMs = DEFAULT_RECOVERY_PROBE_TIMEOUT_MS,
    sessionToolServer,
    nativeDelegationAdapter,
    builtinMcp = builtinMcpServers(),
  } = {}) {
    this.protocol = protocol
    this.root = root
    this.ownership = ownership === 'external' ? 'external' : 'owned'
    this.permissionMode = permissionMode === 'full' ? 'full' : 'native'
    // An empty model means the Agent owns model selection for both new and
    // resumed Sessions. `auto` is retained only as a legacy configuration
    // spelling and is normalized to the same no-override state here.
    this.model = explicitModel(model)
    this.timeoutMs = timeoutMs
    this.directory = directory
    this.baseUrl = clean(baseUrl) || null
    this.coordinatorAgent = clean(coordinatorAgent)
    this.profile = profile || acpBackendProfile({
      protocol,
      root,
      ownership: this.ownership,
      directory,
      cliPath,
      baseUrl,
      token,
      tokenFile,
      coordinatorAgent,
      configDirectory,
      claudeExecutable,
      permissionMode: this.permissionMode,
      model: this.model,
    })
    this.registry = new AcpSessionRegistry({ filePath: sessionStatePath })
    this.sessionToolServer = sessionToolServer || new AcpSessionToolServer()
    // Baseline stdio MCP servers (e.g. open-computer-use) injected into every
    // Session; backends spawn and connect to them on their own.
    this.builtinMcp = this.profile.sessionMcp === false
      ? []
      : (Array.isArray(builtinMcp) ? builtinMcp : [])
    this.builtinMcpLifecycle = !client && clientFactory === createAcpClient
      ? createBuiltinMcpLifecycle(this.builtinMcp)
      : { markUsed() {}, close: () => Promise.resolve() }
    this.activeSessionToolBindings = new WeakMap()
    this.permissionBroker = new PermissionBroker({
      protocol: this.protocol,
      permissionMode: this.permissionMode,
      isTrustedInternalToolRequest: context => (
        this.isTrustedInternalToolRequest(context)
      ),
    })
    this.coordinatorSessions = new Map()
    this.coordinatorSessionPromises = new Map()
    // ACP agents may cache the first MCP connection for a Session. Keep its
    // descriptor stable while serialized owner turns replace the run context.
    this.coordinatorToolRegistrations = new Map()
    this.coordinatorToolRegistrationPromises = new Map()
    this.sessionExecutor = new KeyedSerialExecutor()
    this.activeCoordinatorTurns = new Set()
    // Track every request that reaches the persistent coordinator Session,
    // whether it stays there or delegates to a project Session.
    this.coordinationRuns = new Map()
    this.workControllers = new Map()
    this.workOwners = new Map()
    this.workEventListeners = new Set()
    this.runtimeState = new BackendRuntimeState({
      protocol: this.protocol,
      ownership: this.ownership,
      connectionKind: this.profile.acpConnection?.kind,
      label: this.profile.label,
      stderr: () => this.client?.stderr,
    })
    this.nativeDelegationAdapter = nativeDelegationAdapter || null
    this.backendAvailable = client ? null : backendAvailable
    this.readinessPollMs = readinessPollMs
    this.readinessTimeoutMs = readinessTimeoutMs
    this.recoveryProbeTimeoutMs = Math.max(
      1,
      Number(recoveryProbeTimeoutMs) || DEFAULT_RECOVERY_PROBE_TIMEOUT_MS,
    )
    this.client = client || clientFactory({
      label: this.profile.label,
      connection: this.profile.acpConnection,
      timeoutMs,
      onPermission: (params, context) => (
        this.handlePermission(params, context)
      ),
      sanitizeProcessOutput: this.profile.sanitizeProcessOutput,
      formatRequestError: this.profile.formatRequestError,
    })
  }

  get label() {
    return this.profile.label
  }

  coordinatorUsesMcpInstructions() {
    return this.profile.coordinatorMcpInstructions === true
      && Buffer.byteLength(this.stableCoordinatorInstructions(), 'utf8')
        <= COORDINATOR_MCP_INSTRUCTIONS_MAX_BYTES
  }

  stableCoordinatorInstructions() {
    return [
      COORDINATOR_STABLE_INSTRUCTIONS,
      clean(this.profile.sessionInstructions),
    ].filter(Boolean).join('\n\n')
  }

  get lastHealthFailure() {
    return this.runtimeState.lastFailure
  }

  get runtimeHealth() {
    return this.runtimeState.value
  }

  get pendingPermissions() {
    return this.permissionBroker.pending
  }

  get resolvedPermissions() {
    return this.permissionBroker.resolved
  }

  describe() {
    return {
      kind: this.protocol,
      label: this.label,
      baseUrl: this.baseUrl,
      uiPath: null,
      model: this.model || null,
      directory: this.directory,
      ownership: this.ownership,
      permissionMode: this.permissionMode,
      transport: 'acp',
      acpConnection: this.profile.acpConnection?.kind || null,
      backendAgent: this.coordinatorAgent || null,
      sessionModel: 'one-persistent-backend-agent',
      capabilities: {
        ...this.profile.capabilities,
        taskUpdates: 'activity',
      },
    }
  }

  runtimeStatus() {
    return this.runtimeState.status({ clientReady: this.client.ready })
  }

  status(taskId, { ownerId } = {}) {
    const id = clean(taskId)
    if (!id) return this.runtimeStatus()
    const run = this.coordinationRuns.get(id)
    if (!run || run.ownerId !== clean(ownerId)) {
      return { taskId: id, state: 'not_found', activity: [] }
    }
    const delegation = run.delegation
    if (!delegation) {
      return { taskId: id, state: 'working', activity: [] }
    }
    const current = this.statusForDelegation(
      { delegation_id: delegation.id },
      run.ownerId,
    )
    return {
      taskId: id,
      state: current.status === 'running' ? 'working' : current.status,
      activity: current.recent_updates || [],
      ...(current.result ? { result: current.result } : {}),
      ...(current.error ? { error: current.error } : {}),
    }
  }

  markRuntimeReady(initialized) {
    this.runtimeState.ready(initialized)
  }

  markRuntimeFailure(error) {
    this.runtimeState.failed(error)
  }

  async start({ signal } = {}) {
    // Injected ACP clients used by embedders may already be ready and expose
    // only Session operations. The adapter itself still satisfies BackendPort.
    if (typeof this.client.start !== 'function') return this.runtimeStatus()
    try {
      await this.waitForBackendReadiness(signal)
      this.runtimeState.starting()
      const initialized = await this.client.start()
      if (this.profile.externalMcp) {
        assertMcpServerCapabilities({
          label: this.label,
          capabilities: initialized?.agentCapabilities,
          mcpServers: [{ type: 'http' }],
        })
      }
      this.markRuntimeReady(initialized)
      return this.runtimeStatus()
    } catch (error) {
      if (!signal?.aborted) this.markRuntimeFailure(error)
      throw error
    }
  }

  async health() {
    if (
      this.runtimeState.shouldBackoff()
    ) {
      return this.runtimeStatus()
    }
    try {
      if (
        this.profile.readinessMessage
        && this.baseUrl
        && this.backendAvailable
        && !await this.backendAvailable(this.baseUrl)
      ) {
        this.runtimeState.waiting(this.profile.readinessMessage)
        return this.runtimeStatus()
      }
      this.runtimeState.starting()
      const initialized = await this.client.start()
      if (this.profile.externalMcp) {
        assertMcpServerCapabilities({
          label: this.label,
          capabilities: initialized?.agentCapabilities,
          mcpServers: [{ type: 'http' }],
        })
      }
      this.markRuntimeReady(initialized)
      return this.runtimeStatus()
    } catch (error) {
      this.markRuntimeFailure(error)
      return this.runtimeStatus()
    }
  }

  needsBackendReadinessProbe() {
    return Boolean(
      this.profile.readinessMessage
      && this.baseUrl
      && this.backendAvailable,
    )
  }

  async waitForBackendReadiness(signal) {
    if (!this.needsBackendReadinessProbe()) return
    if (signal?.aborted) {
      throw signal.reason || new Error('任务已取消')
    }
    const deadline = Date.now() + this.readinessTimeoutMs
    while (!await this.backendAvailable(this.baseUrl)) {
      this.runtimeState.waiting(this.profile.readinessMessage)
      if (Date.now() >= deadline) {
        const error = new Error(
          `${this.profile.label} 后台服务启动超时`,
        )
        error.code = 'ETIMEDOUT'
        throw error
      }
      await waitForRetry(this.readinessPollMs, signal)
    }
  }

  serialize(key, operation) {
    return this.sessionExecutor.run(key, operation)
  }

  publishWorkEvent(event, { taskId, ownerId, onEvent } = {}) {
    try {
      onEvent?.(event)
    } catch {
      // A per-submission observer must not break backend execution.
    }
    const published = {
      ...event,
      taskId: clean(taskId) || null,
      ownerId: clean(ownerId) || null,
    }
    for (const listener of this.workEventListeners) {
      try {
        listener(published)
      } catch {
        // Subscribers are isolated from the adapter and from each other.
      }
    }
  }

  subscribe(listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('BackendPort subscriber must be a function')
    }
    this.workEventListeners.add(listener)
    return () => this.workEventListeners.delete(listener)
  }

  async ensureCoordinatorSession(ownerId, mcpServers = []) {
    if (this.builtinMcp.length) this.builtinMcpLifecycle.markUsed()
    const key = coordinatorKey(ownerId, this.protocol)
    if (this.coordinatorSessions.has(key)) {
      return this.coordinatorSessions.get(key)
    }
    if (this.coordinatorSessionPromises.has(key)) {
      return this.coordinatorSessionPromises.get(key)
    }
    const pending = (async () => {
      const stored = this.registry.get(key, {
        ownerId,
        // Legacy coordinator records are safe to migrate because their
        // storage key has always contained the encoded owner identity.
        allowUnowned: true,
      })
      const contractVersion = COORDINATOR_CONTRACT_VERSION
      let session
      const canResumeStored = Boolean(
        stored?.sessionId
        && stored.contractVersion === contractVersion,
      )
      if (stored?.sessionId && !canResumeStored) {
        this.registry.delete(key)
      }
      if (canResumeStored) {
        try {
          session = await this.client.resumeSession(stored.sessionId, {
            cwd: stored.cwd || this.directory,
            mcpServers,
            meta: this.coordinatorMeta(ownerId),
            ownerId,
            role: 'coordinator',
          })
          const accepted = this.profile.acceptResumedSession?.({
            session,
            role: 'coordinator',
            coordinatorAgent: this.coordinatorAgent,
          }) !== false
          if (!accepted) {
            this.registry.delete(key)
            session = null
          } else {
            session.isNew = false
          }
        } catch {
          this.registry.delete(key)
        }
      }
      if (!session) {
        session = await this.client.newSession({
          cwd: this.directory,
          mcpServers,
          meta: this.coordinatorMeta(ownerId),
          ownerId,
          role: 'coordinator',
        })
        session.isNew = true
      }
      session.ownerId = clean(ownerId)
      session.contractVersion = contractVersion
      await this.configureSession(session, 'coordinator')
      this.coordinatorSessions.set(key, session)
      this.registry.set(key, session, { ownerId })
      return session
    })().finally(() => this.coordinatorSessionPromises.delete(key))
    this.coordinatorSessionPromises.set(key, pending)
    return pending
  }

  async ensureCoordinatorToolRegistration(ownerId, context) {
    if (!this.profile.externalMcp) return null
    const key = coordinatorKey(ownerId, this.protocol)
    const current = this.coordinatorToolRegistrations.get(key)
    if (current) {
      current.update(context)
      return current
    }
    const existingPromise = this.coordinatorToolRegistrationPromises.get(key)
    if (existingPromise) {
      const registration = await existingPromise
      registration.update(context)
      return registration
    }
    const pending = this.sessionToolServer.register(context, {
      instructions: this.coordinatorUsesMcpInstructions()
        ? this.stableCoordinatorInstructions()
        : '',
    }).then(
      registration => {
        this.coordinatorToolRegistrations.set(key, registration)
        return registration
      },
    ).finally(() => {
      this.coordinatorToolRegistrationPromises.delete(key)
    })
    this.coordinatorToolRegistrationPromises.set(key, pending)
    return pending
  }

  isTrustedInternalToolRequest({ params, session, toolName } = {}) {
    if (!toolName || !session || typeof session !== 'object') return false
    const binding = this.activeSessionToolBindings.get(session)
    if (!binding) return false
    const ownerId = clean(session.ownerId)
    const sessionId = clean(session.sessionId)
    const runId = clean(session.coordinationRunId)
    const permissionScopeId = clean(session.permissionScopeId)
    if (
      !ownerId
      || !sessionId
      || !runId
      || !permissionScopeId
      || clean(params?.sessionId) !== sessionId
      || session.role !== 'coordinator'
      || binding.ownerId !== ownerId
      || binding.sessionId !== sessionId
      || binding.runId !== runId
      || binding.permissionScopeId !== permissionScopeId
      || binding.run !== this.coordinationRuns.get(runId)
      || binding.run.ownerId !== ownerId
      || binding.run.sessionId !== sessionId
      || !this.activeCoordinatorTurns.has(sessionId)
    ) {
      return false
    }
    const registration = this.coordinatorToolRegistrations.get(
      coordinatorKey(ownerId, this.protocol),
    )
    if (registration !== binding.registration) return false
    return Array.isArray(session.mcpServers)
      && session.mcpServers.includes(registration.descriptor)
  }

  coordinatorMeta(ownerId) {
    return this.profile.coordinatorMeta?.(ownerId) || null
  }

  async configureSession(session, role) {
    let options = Array.isArray(session?.response?.configOptions)
      ? session.response.configOptions
      : []
    if (role === 'coordinator' && this.coordinatorAgent) {
      const configId = 'mode'
      const value = this.coordinatorAgent
      const option = options.find(item => clean(item?.id) === configId)
      const supported = option?.type !== 'select'
        || option.options?.some(item => item.value === value)
      if (option && supported && option.currentValue !== value) {
        await this.client.setSessionConfigOption(
          session.sessionId,
          configId,
          value,
        )
        option.currentValue = value
      }
    }
    options = await this.applyProfileSessionConfig(session, options)
    if (this.model && this.profile.processModelConfiguration !== true) {
      await this.forceSessionModel(session, options)
    }
  }

  async applyProfileSessionConfig(session, initialOptions) {
    let options = initialOptions
    for (const setting of this.profile.sessionConfigOptions || []) {
      const id = clean(setting?.id)
      const desired = clean(setting?.value)
      const option = options.find(item => clean(item?.id) === id)
      const selected = option?.type === 'select'
        ? matchingOptionValue(option.options, desired)
        : desired
      if (!option || !selected) {
        throw new AgentError(
          `${this.label} 没有通过 ACP 提供必要的 Session 配置 ${id}=${desired}`,
          { status: 422, protocol: 'acp' },
        )
      }
      if (modelKey(option.currentValue) === modelKey(selected)) continue
      let response
      try {
        response = await this.client.setSessionConfigOption(
          session.sessionId,
          id,
          selected,
        )
      } catch (error) {
        throw new AgentError(
          `${this.label} 无法设置 Session 配置 ${id}=${desired}：${
            clean(error?.message) || '未知错误'
          }`,
          { status: error.status || 502, protocol: 'acp' },
        )
      }
      const updatedOptions = Array.isArray(response?.configOptions)
        ? response.configOptions
        : null
      const updated = updatedOptions?.find(item => clean(item?.id) === id)
      if (modelKey(updated?.currentValue) !== modelKey(selected)) {
        throw new AgentError(
          `${this.label} 未确认 Session 配置生效：要求 ${id}=${desired}，实际 ${
            clean(updated?.currentValue) || '未知'
          }`,
          { status: 502, protocol: 'acp' },
        )
      }
      options = updatedOptions
      session.response = {
        ...(session.response || {}),
        configOptions: options,
      }
    }
    return options
  }

  async forceSessionModel(session, options) {
    const desired = this.model
    const option = modelConfigOption(options)
    if (!option) {
      if (this.nativeDelegationAdapter?.setSessionModel) {
        try {
          await this.nativeDelegationAdapter.setSessionModel({
            sessionKey: clean(session?.meta?.sessionKey)
              || clean(session?.sessionId),
            model: desired,
          })
          return
        } catch (error) {
          throw new AgentError(
            `${this.label} 无法把 Session 模型设置为 ${desired}：${
              clean(error?.message) || '未知错误'
            }`,
            {
              status: error.status || 502,
              protocol: error.protocol || `${this.protocol}-native`,
            },
          )
        }
      }
      const legacy = legacyModelState(session?.response)
      if (legacy && this.client.setLegacySessionModel) {
        const selected = matchingOptionValue(
          legacy.choices.map(choice => ({
            value: choice.value,
            name: choice.names[0],
          })),
          desired,
        )
        if (!selected) {
          const availableModels = legacy.choices.map(choice => (
            choice.names[0] || choice.value
          ))
          const available = availableModels.length
            ? `；可选模型：${availableModels.slice(0, 12).join('、')}`
            : ''
          throw new AgentError(
            `${this.label} 当前 Session 不支持模型 ${desired}${available}`,
            { status: 422, protocol: 'acp' },
          )
        }
        if (modelKey(legacy.currentValue) === modelKey(selected)) return
        try {
          await this.client.setLegacySessionModel(
            session.sessionId,
            selected,
          )
        } catch (error) {
          throw new AgentError(
            `${this.label} 无法把 Session 模型设置为 ${desired}：${
              clean(error?.message) || '未知错误'
            }`,
            {
              status: error.status || 502,
              protocol: 'acp',
            },
          )
        }
        session.response.models.currentModelId = selected
        return
      }
      throw new AgentError(
        `${this.label} 没有通过 ACP 提供 Session 模型配置，`
        + `无法强制使用模型 ${desired}`,
        { status: 422, protocol: 'acp' },
      )
    }
    const values = optionValues(option.options)
    const selected = option.type === 'select'
      ? matchingOptionValue(option.options, desired)
      : desired
    if (option.type === 'select' && !selected) {
      const available = values.length
        ? `；可选模型：${values.slice(0, 12).join('、')}`
        : ''
      throw new AgentError(
        `${this.label} 当前 Session 不支持模型 ${desired}${available}`,
        { status: 422, protocol: 'acp' },
      )
    }
    if (modelKey(option.currentValue) === modelKey(selected)) return
    let response
    try {
      response = await this.client.setSessionConfigOption(
        session.sessionId,
        option.id,
        selected,
      )
    } catch (error) {
      throw new AgentError(
        `${this.label} 无法把 Session 模型设置为 ${desired}：${
          clean(error?.message) || '未知错误'
        }`,
        {
          status: error.status || 502,
          protocol: 'acp',
        },
      )
    }
    const updatedOptions = Array.isArray(response?.configOptions)
      ? response.configOptions
      : null
    if (!updatedOptions) {
      throw new AgentError(
        `${this.label} 设置模型后没有返回 ACP configOptions，`
        + `无法确认模型 ${desired} 已生效`,
        { status: 502, protocol: 'acp' },
      )
    }
    const updated = updatedOptions.find(item => (
      clean(item?.id) === clean(option.id)
    ))
      || modelConfigOption(updatedOptions)
    if (modelKey(updated?.currentValue) !== modelKey(selected)) {
      throw new AgentError(
        `${this.label} 未确认模型覆盖生效：要求 ${desired}，`
        + `实际 ${clean(updated?.currentValue) || '未知'}`,
        { status: 502, protocol: 'acp' },
      )
    }
    session.response = {
      ...(session.response || {}),
      configOptions: updatedOptions,
    }
  }

  async handlePermission(params, { signal, session } = {}) {
    return this.permissionBroker.request(params, { signal, session })
  }

  cancelPermission(record) {
    return this.permissionBroker.cancel(record)
  }

  async resolveAuthorization(id, decision, { ownerId } = {}) {
    return this.permissionBroker.respond(id, decision, { ownerId })
  }

  cancelPermissionsForScope(permissionScopeId) {
    this.permissionBroker.cancelScope(permissionScopeId)
  }

  rememberDelegationUpdate(record, activity) {
    if (
      !record
      || !activity
      || !['plan', 'tool', 'thinking', 'mode', 'session'].includes(activity.kind)
    ) {
      return
    }
    const key = clean(activity.id)
      || `${activity.kind}:${clean(activity.tool)}:${clean(activity.detail)}`
    const summary = Object.fromEntries(Object.entries(activity)
      .filter(([name, value]) => (
        name !== 'id'
        && value !== ''
        && value !== null
        && value !== undefined
      )))
    record.recentUpdates ||= []
    record.recentUpdates = record.recentUpdates
      .filter(item => item.key !== key)
    record.recentUpdates.push({ key, summary })
    record.recentUpdates = record.recentUpdates
      .slice(-MAX_DELEGATION_RECENT_UPDATES)
  }

  onSessionUpdate(run, update, { delegation = null } = {}) {
    run.receivedUpdate = true
    run.toolCalls ||= new Map()
    run.messageStreams ||= {}
    const streamedMessage = messageFromUpdate(update, run.messageStreams)
    if (streamedMessage?.message) {
      run.onEvent?.(backendEvent(BackendEventType.MESSAGE, {
        ...streamedMessage,
        streaming: true,
      }))
    }
    const activity = activityFromUpdate(update, run.toolCalls)
    if (activity) run.onEvent?.(backendEvent(BackendEventType.ACTIVITY, { activity }))
    if (delegation && this.profile.externalMcp) {
      this.rememberDelegationUpdate(delegation, activity)
    }
    if (!this.profile.nativeDelegation) return
    if (!['tool_call', 'tool_call_update'].includes(update?.sessionUpdate)) {
      return
    }
    const id = clean(update.toolCallId)
    const merged = {
      ...(run.nativeToolCalls.get(id) || {}),
      ...update,
    }
    run.nativeToolCalls.set(id, merged)
    if (merged.status !== 'completed' || run.delegation) return
    const name = clean(merged.name || merged.title).toLowerCase()
    if (!/sessions_(spawn|send)/.test(name)) return
    const output = nativeToolOutput(merged.rawOutput)
    const sessionId = clean(
      output.childSessionKey
      || output.sessionKey
      || output.session_id
      || output.sessionId,
    )
    if (!sessionId) return
    const delegationId = clean(output.runId)
    if (!delegationId) return
    const directory = clean(
      merged.rawInput?.cwd
      || merged.rawInput?.directory
      || this.directory,
    )
    const title = bounded(
      merged.rawInput?.label
      || merged.rawInput?.task
      || merged.rawInput?.message,
      160,
    ) || this.profile.defaultDelegationTitle || `${this.label} 项目任务`
    this.rememberProjectSession({
      sessionId,
      cwd: directory,
      title,
    }, run.ownerId)
    run.delegation = this.createNativeDelegation(run, {
      delegationId,
      sessionId,
      directory,
      title,
    })
  }

  projectSessionRecord(ownerId, sessionId, { backendSession = null } = {}) {
    const owner = clean(ownerId)
    const id = clean(sessionId)
    if (!owner || !id) return null
    const key = projectSessionKey(owner, this.protocol, id)
    let record = this.registry.getProject(key, {
      ownerId: owner,
      // The owner is encoded into this key, so an intermediate record written
      // before ownerId became an explicit field remains attributable.
      allowUnowned: true,
    })
    if (record) {
      if (!clean(record.ownerId)) {
        this.registry.setProject(key, record, { ownerId: owner })
        record = this.registry.getProject(key, { ownerId: owner })
      }
      return record
    }

    const legacyKey = legacyProjectSessionKey(this.protocol, id)
    const legacy = this.registry.getProject(legacyKey)
    const provenBackendOwner = sessionOwnerId(backendSession)
    if (!legacy) return null
    const legacyOwner = clean(legacy.ownerId)
    // Conflicting durable ownership is never overwritten by fresh global
    // metadata. An unowned legacy record may migrate only when the backend
    // independently attributes that exact Session to the current owner.
    if (legacyOwner ? legacyOwner !== owner : provenBackendOwner !== owner) {
      return null
    }

    const migrated = {
      ...legacy,
      ...(backendSession || {}),
      sessionId: id,
      cwd: clean(backendSession?.cwd || legacy.cwd),
      title: clean(backendSession?.title || legacy.title),
    }
    this.registry.setProject(key, migrated, { ownerId: owner })
    this.registry.deleteProject(legacyKey)
    return this.registry.getProject(key, { ownerId: owner })
  }

  ownsProjectSession(session, ownerId) {
    const owner = clean(ownerId)
    const id = clean(session?.sessionId)
    if (!owner || !id) return false
    const backendOwner = sessionOwnerId(session)
    if (backendOwner && backendOwner !== owner) return false
    if (this.projectSessionRecord(owner, id, { backendSession: session })) {
      return true
    }
    if (backendOwner !== owner) return false
    const legacy = this.registry.getProject(
      legacyProjectSessionKey(this.protocol, id),
    )
    if (clean(legacy?.ownerId) && clean(legacy.ownerId) !== owner) {
      return false
    }
    this.rememberProjectSession(session, owner)
    return true
  }

  async listProjectSessions(ownerId, { query, limit } = {}) {
    const owner = clean(ownerId)
    if (!owner) return { sessions: [] }
    const sessions = await this.client.listSessions({
      limit: limit || 20,
    })
    const ownedSessions = sessions.filter(session => (
      clean(session?.sessionId)
      && clean(session?.cwd)
      && this.ownsProjectSession(session, owner)
    ))
    const coordinators = new Set(
      [...this.coordinatorSessions.values()].map(item => item.sessionId),
    )
    const storedCoordinator = this.registry.get(
      coordinatorKey(owner, this.protocol),
      { ownerId: owner, allowUnowned: true },
    )
    if (storedCoordinator?.sessionId) {
      coordinators.add(clean(storedCoordinator.sessionId))
    }
    const needle = clean(query).toLowerCase()
    return {
      sessions: ownedSessions
        .filter(session => !coordinators.has(clean(session.sessionId)))
        .map(sessionSummary)
        .filter(session => !needle || [
          session.title,
          session.directory,
        ].join(' ').toLowerCase().includes(needle)),
    }
  }

  rememberProjectSession(session, ownerId) {
    const owner = clean(ownerId || sessionOwnerId(session))
    if (!owner || !clean(session?.sessionId) || !clean(session?.cwd)) return
    const declaredOwner = sessionOwnerId(session)
    if (declaredOwner && declaredOwner !== owner) {
      throw new AgentError(`${this.label} Session 所有者与当前用户不匹配`, {
        status: 404,
        protocol: this.protocol,
      })
    }
    const legacy = this.registry.getProject(
      legacyProjectSessionKey(this.protocol, session.sessionId),
    )
    if (clean(legacy?.ownerId) && clean(legacy.ownerId) !== owner) {
      throw new AgentError(`${this.label} Session 已属于其他用户`, {
        status: 404,
        protocol: this.protocol,
      })
    }
    this.registry.setProject(
      projectSessionKey(owner, this.protocol, session.sessionId),
      session,
      { ownerId: owner },
    )
  }

  findDelegation({ delegation_id: delegationId, session_id: sessionId }, ownerId) {
    const owner = clean(ownerId)
    if (!owner) return undefined
    return [...this.coordinationRuns.values()]
      .map(run => run.delegation)
      .find(record => (
        record?.ownerId === owner
        && (
          (clean(delegationId) && record?.id === clean(delegationId))
          || (clean(sessionId) && record?.sessionId === clean(sessionId))
        )
      ))
  }

  createDelegation(run, {
    session,
    prompt,
    directory,
    title,
  }) {
    if (run.delegation) {
      throw new AgentError('当前协调轮次已经启动了一个第三层任务', {
        protocol: this.protocol,
      })
    }
    const controller = new AbortController()
    const record = {
      id: `${this.protocol}_run_${randomUUID()}`,
      sessionId: session.sessionId,
      directory,
      title: bounded(title || prompt, 160) || `${this.label} 项目任务`,
      ownerId: run.ownerId,
      taskId: run.coordinationRunId,
      status: 'running',
      controller,
      recentUpdates: [],
      result: null,
      error: null,
    }
    run.delegation = record
    record.promise = this.serialize(`target:${record.sessionId}`, async () => {
      const permissionScopeId = `prompt_${randomUUID()}`
      try {
        session.ownerId = run.ownerId
        session.coordinationRunId = run.coordinationRunId
        session.onEvent = run.onEvent
        session.permissionScopeId = permissionScopeId
        const result = assertCompletedAcpTurn(await this.client.prompt(
          record.sessionId,
          appendPromptBlocks(prompt, run.inputBlocks),
          {
            signal: controller.signal,
            timeoutMs: 0,
            onUpdate: update => this.onSessionUpdate(run, update, {
              delegation: record,
            }),
          },
        ), {
          signal: controller.signal,
          label: `${this.label} 项目 Session`,
          protocol: this.protocol,
        })
        record.status = 'completed'
        record.result = result
        return {
          id: record.id,
          sessionId: record.sessionId,
          directory: record.directory,
          title: record.title,
          content: result.content,
          contentBlocks: result.contentBlocks || [],
        }
      } catch (error) {
        record.status = controller.signal.aborted ? 'cancelled' : 'failed'
        record.error = error
        throw error
      } finally {
        this.cancelPermissionsForScope(permissionScopeId)
        if (session.permissionScopeId === permissionScopeId) {
          session.permissionScopeId = null
        }
      }
    })
    record.promise.catch(() => {})
    return record
  }

  async startProjectSession(run, { prompt, title }) {
    const cwd = clean(run.cwd) || this.directory
    const session = await this.client.newSession({
      cwd,
      mcpServers: this.builtinMcp,
      ownerId: run.ownerId,
      role: 'project',
    })
    this.rememberProjectSession({
      ...session,
      cwd,
      title: clean(title || prompt),
    }, run.ownerId)
    await this.configureSession(session, 'project')
    const record = this.createDelegation(run, {
      session,
      prompt: clean(prompt),
      directory: cwd,
      title,
    })
    return {
      status: 'started',
      delegation_id: record.id,
      session_id: record.sessionId,
      title: record.title,
      directory: record.directory,
    }
  }

  async continueProjectSession(
    run,
    { session_id: sessionId, prompt },
  ) {
    const active = this.findDelegation({ session_id: sessionId }, run.ownerId)
    let remembered = this.projectSessionRecord(run.ownerId, sessionId)
    let existing = null
    let cwd = clean(
      active?.directory
      || remembered?.cwd,
    )
    if (!cwd) {
      const sessions = await this.client.listSessions({
        limit: MAX_SESSION_RESULTS,
      })
      existing = sessions.find(item => (
        clean(item.sessionId) === clean(sessionId)
        && this.ownsProjectSession(item, run.ownerId)
      ))
      remembered = this.projectSessionRecord(run.ownerId, sessionId, {
        backendSession: existing,
      })
      cwd = clean(existing?.cwd)
    }
    if (!cwd) {
      throw new AgentError(
        `${this.label} Session 的项目目录未知，请先查询 Session 列表后再继续`,
        { protocol: this.protocol },
      )
    }
    const session = await this.client.resumeSession(clean(sessionId), {
      cwd,
      mcpServers: this.builtinMcp,
      ownerId: run.ownerId,
      role: 'project',
    })
    this.rememberProjectSession({
      ...existing,
      ...session,
      cwd,
      title: existing?.title || remembered?.title,
    }, run.ownerId)
    await this.configureSession(session, 'project')
    const record = this.createDelegation(run, {
      session,
      prompt: clean(prompt),
      directory: cwd,
      title: clean(prompt),
    })
    return {
      status: 'started',
      delegation_id: record.id,
      session_id: record.sessionId,
      title: record.title,
      directory: record.directory,
    }
  }

  statusForDelegation(input, ownerId) {
    const record = this.findDelegation(input, ownerId)
    if (!record) return { status: 'not_found' }
    return {
      status: record.status,
      delegation_id: record.id,
      session_id: record.sessionId,
      title: record.title,
      directory: record.directory,
      ...(Array.isArray(record.recentUpdates)
        ? { recent_updates: record.recentUpdates.map(item => item.summary) }
        : {}),
      ...(record.status === 'completed'
        ? { result: clean(record.result?.content).slice(0, 4000) }
        : {}),
      ...(record.status === 'failed'
        ? { error: clean(record.error?.message || record.error) }
        : {}),
    }
  }

  async cancelDelegation(input, ownerId) {
    const record = this.findDelegation(input, ownerId)
    if (!record) return { status: 'not_found' }
    if (!['completed', 'failed', 'cancelled'].includes(record.status)) {
      record.status = 'cancelled'
      if (this.nativeDelegationAdapter) {
        await this.nativeDelegationAdapter.cancel({
          runId: record.id,
          sessionId: record.sessionId,
        }).catch(() => {})
      }
      record.controller.abort(new Error('用户已取消这项项目任务'))
      if (!this.nativeDelegationAdapter) {
        await this.client.cancelSession(record.sessionId).catch(() => {})
      }
    }
    return {
      status: record.status,
      delegation_id: record.id,
      session_id: record.sessionId,
    }
  }

  toolContext(run) {
    return {
      listSessions: input => this.listProjectSessions(run.ownerId, input),
      startSession: input => this.startProjectSession(run, input),
      sendSession: input => this.continueProjectSession(run, input),
      sessionStatus: input => this.statusForDelegation(input, run.ownerId),
      cancelSession: input => this.cancelDelegation(input, run.ownerId),
    }
  }

  coordinatorInstructions(message) {
    if (this.coordinatorUsesMcpInstructions()) return message
    const sessionInstructions = clean(this.profile.sessionInstructions)
    if (!sessionInstructions) return message
    return transformPromptText(message, content => [
      '<qwen_audio_agent_backend_profile>',
      sessionInstructions,
      '</qwen_audio_agent_backend_profile>',
      '',
      content,
    ].join('\n'))
  }

  async promptCoordinator(session, prompt, run, {
    signal,
    onUpdate,
  } = {}) {
    let attempt = 0
    while (true) {
      try {
        return assertCompletedAcpTurn(await this.client.prompt(
          session.sessionId,
          this.coordinatorInstructions(prompt),
          {
            signal,
            // Agent turns are user-cancellable and may legitimately run for
            // hours. Connection setup and control RPCs remain bounded, but a
            // live coding turn has no artificial wall-clock deadline.
            timeoutMs: 0,
            onUpdate,
          },
        ), {
          signal,
          label: `${this.label} 协调 Session`,
          protocol: this.protocol,
        })
      } catch (error) {
        const retryIsSafe = !run.receivedUpdate
          && !run.delegation
          && run.nativeToolCalls.size === 0
          && run.toolCalls.size === 0
        const delayMs = retryIsSafe
          ? this.profile.promptRetryDelay?.({ error, attempt })
          : null
        if (!Number.isFinite(delayMs) || delayMs < 0) throw error
        attempt += 1
        await waitForRetry(delayMs, signal)
      }
    }
  }

  async coordinatorTurn(message, {
    ownerId,
    coordinationRunId,
    coordinationRequestId,
    signal,
    onEvent,
  }) {
    const run = {
      ownerId: clean(ownerId),
      coordinationRunId: clean(coordinationRunId),
      coordinationRequestId: clean(coordinationRequestId)
        || clean(coordinationRunId),
      onEvent,
      delegation: null,
      nativeToolCalls: new Map(),
      toolCalls: new Map(),
      initialPromptDone: false,
      receivedUpdate: false,
      inputBlocks: nonTextPromptBlocks(message),
    }
    if (run.coordinationRunId) {
      this.coordinationRuns.set(run.coordinationRunId, run)
    }
    const key = coordinatorKey(ownerId, this.protocol)
    const pendingFacts = this.registry.reconciliationsFor(key)
    const prompt = pendingFacts.length
      ? transformPromptText(message, content => [
          '上一请求已取消，不要续接其未完成内容。仅处理以下新请求。',
          '',
          content,
        ].join('\n'))
      : message
    const registration = await this.ensureCoordinatorToolRegistration(
      ownerId,
      this.toolContext(run),
    )
    const mcpServers = [
      ...(registration ? [registration.descriptor] : []),
      ...this.builtinMcp,
    ]
    const session = await this.ensureCoordinatorSession(ownerId, mcpServers)
    const permissionScopeId = `prompt_${randomUUID()}`
    run.sessionId = session.sessionId
    run.cwd = clean(session.cwd) || this.directory
    session.ownerId = clean(ownerId)
    session.role = 'coordinator'
    session.coordinationRunId = clean(coordinationRunId)
    session.onEvent = onEvent
    session.permissionScopeId = permissionScopeId
    this.activeCoordinatorTurns.add(session.sessionId)
    const internalToolBinding = registration
      ? {
          registration,
          run,
          ownerId: run.ownerId,
          sessionId: clean(session.sessionId),
          runId: run.coordinationRunId,
          permissionScopeId,
        }
      : null
    if (internalToolBinding) {
      this.activeSessionToolBindings.set(session, internalToolBinding)
    }
    try {
      // Re-supply MCP definitions on resume: ACP Sessions do not require the
      // agent to persist client-provided MCP connections across processes.
      if (mcpServers.length && !session.isNew) {
        await this.client.resumeSession(session.sessionId, {
          cwd: session.cwd || this.directory,
          mcpServers,
          meta: this.coordinatorMeta(ownerId),
          ownerId,
          role: 'coordinator',
        })
      }
      const result = await this.promptCoordinator(session, prompt, run, {
        signal,
        onUpdate: update => this.onSessionUpdate(run, update),
      })
      if (
        !clean(result?.content)
        && !(result?.contentBlocks || []).some(block => block?.type !== 'text')
      ) {
        const error = new AgentError(
          `${this.profile.label} ACP Session 未返回任何内容`,
          { status: 502, protocol: this.protocol },
        )
        error.recoverWithFreshCoordinator = !run.receivedUpdate
          && !run.delegation
          && run.nativeToolCalls.size === 0
          && run.toolCalls.size === 0
        throw error
      }
      run.initialPromptDone = true
      session.isNew = false
      if (pendingFacts.length) {
        this.registry.acknowledgeReconciliations(key, pendingFacts)
      }
      this.registry.set(
        coordinatorKey(ownerId, this.protocol),
        session,
        { ownerId },
      )
      return {
        run,
        session,
        result,
      }
    } finally {
      if (
        internalToolBinding
        && this.activeSessionToolBindings.get(session) === internalToolBinding
      ) {
        this.activeSessionToolBindings.delete(session)
      }
      this.activeCoordinatorTurns.delete(session.sessionId)
      this.cancelPermissionsForScope(permissionScopeId)
      if (session.permissionScopeId === permissionScopeId) {
        session.permissionScopeId = null
      }
    }
  }

  createNativeDelegation(run, {
    delegationId,
    sessionId,
    directory,
    title,
  }) {
    const controller = new AbortController()
    const record = {
      id: delegationId,
      sessionId,
      directory,
      title,
      ownerId: run.ownerId,
      taskId: run.coordinationRunId,
      status: 'running',
      controller,
      result: null,
      error: null,
      parentSessionId: run.sessionId,
      nativeCompletion: deferred(),
    }
    record.promise = this.waitForNativeDelegation(record, run)
    record.promise.catch(() => {})
    return record
  }

  async waitForNativeDelegation(record) {
    try {
      const completed = this.nativeDelegationAdapter
        ? await this.nativeDelegationAdapter.wait({
            runId: record.id,
            sessionId: record.sessionId,
            signal: record.controller.signal,
          })
        : { content: await record.nativeCompletion.promise }
      const content = clean(completed?.content)
      record.status = 'completed'
      record.result = {
        content,
        contentBlocks: completed?.contentBlocks || [],
      }
      return {
        id: record.id,
        sessionId: record.sessionId,
        directory: record.directory,
        title: record.title,
        content,
        contentBlocks: completed?.contentBlocks || [],
      }
    } catch (error) {
      record.status = record.controller.signal.aborted ? 'cancelled' : 'failed'
      record.error = error
      throw error
    }
  }

  delegationResultPrompt(result, objective = '') {
    const task = clean(objective || result.title).slice(0, 2_000)
    const instruction = [
      '刚才交给独立任务处理的工作已经完成。以下是与当前请求关联的可信最终结果：',
      ...(task ? ['', `原任务：${task}`] : []),
      '',
      clean(result.content).slice(0, MAX_DELEGATION_RESULT_CHARS),
      '',
      '请只整理以上可信结果，直接给出自然的最终答复；',
      '不要输出任务状态或展示协议，也不要再次执行、委托或查询目标任务。',
    ].join('\n')
    const nonTextBlocks = (result.contentBlocks || [])
      .filter(block => block?.type !== 'text')
    return appendPromptBlocks(instruction, nonTextBlocks)
  }

  resultEnvelope(initial, delegation = null, priorContentBlocks = []) {
    return {
      content: initial.result.content,
      contentBlocks: [
        ...(priorContentBlocks || []),
        ...(delegation?.contentBlocks || []),
        ...(initial.result.contentBlocks || []),
      ],
      raw: initial.result.response,
      protocol: this.protocol,
      metadata: {
        backendRef: {
          provider: this.protocol,
          role: 'backend',
          sessionId: initial.session.sessionId,
          directory: initial.session.cwd || this.directory,
        },
        ...(delegation
          ? {
              delegation: {
                id: delegation.id,
                sessionId: delegation.sessionId,
                title: delegation.title,
                directory: delegation.directory,
              },
            }
          : {}),
      },
    }
  }

  discardCoordinatorSession(ownerId, sessionId = '') {
    const key = coordinatorKey(ownerId, this.protocol)
    const current = this.coordinatorSessions.get(key)
    if (!sessionId || current?.sessionId === sessionId) {
      this.coordinatorSessions.delete(key)
    }
    this.registry.delete(key)
  }

  async coordinatorTurnWithRecovery(message, options = {}) {
    try {
      return await this.coordinatorTurn(message, options)
    } catch (error) {
      if (!error?.recoverWithFreshCoordinator) throw error
      this.discardCoordinatorSession(options.ownerId)
      return this.coordinatorTurn(message, options)
    }
  }

  async submit(work, { signal, onEvent } = {}) {
    const taskId = clean(work?.id)
    const ownerId = clean(work?.ownerId)
    const objective = clean(work?.objective ?? work?.message)
    if (!taskId || !ownerId || !objective) {
      throw new AgentError('BackendPort submit requires task id, owner and input', {
        status: 400,
        protocol: this.protocol,
      })
    }
    if (this.workControllers.has(taskId)) {
      throw new AgentError('BackendPort Task is already active', {
        status: 409,
        protocol: this.protocol,
      })
    }
    const controller = new AbortController()
    const workSignal = signal
      ? AbortSignal.any([signal, controller.signal])
      : controller.signal
    this.workControllers.set(taskId, controller)
    this.workOwners.set(taskId, ownerId)
    try {
      const prompt = buildAcpCoordinatorInstruction({
        ...work,
        objective,
        includeStableInstructions: !this.coordinatorUsesMcpInstructions(),
      })
      const run = message => this.runCoordinator(message, {
        ownerId,
        coordinationRunId: taskId,
        coordinationRequestId: taskId,
        workObjective: objective,
        signal: workSignal,
        onEvent,
      })
      const result = await run(promptWithInputParts(prompt, work?.inputParts))
      const artifacts = artifactsFromAcpContentBlocks(result?.contentBlocks)
      if (!clean(result?.content) && !artifacts.length) {
        throw new AgentError('Coordinator backend returned an empty response', {
          status: 502,
          protocol: this.protocol,
        })
      }
      return {
        content: clean(result?.content),
        artifacts,
        presentation: null,
      }
    } finally {
      if (this.workControllers.get(taskId) === controller) {
        this.workControllers.delete(taskId)
        this.workOwners.delete(taskId)
      }
    }
  }

  async runCoordinator(message, {
    ownerId,
    coordinationRunId,
    coordinationRequestId,
    workObjective,
    signal,
    onEvent,
  } = {}) {
    // Health polling and task dispatch share the ACP client's start promise.
    // The execution path additionally waits for an owned service endpoint, so
    // the first task after a cold start cannot race its bridge.
    await this.start({ signal })
    const runId = clean(coordinationRunId)
    const key = coordinatorKey(ownerId, this.protocol)
    const publish = event => this.publishWorkEvent(event, {
      taskId: runId,
      ownerId,
      onEvent,
    })
    try {
      const initial = await this.serialize(
        `coordinator:${key}`,
        () => this.coordinatorTurnWithRecovery(message, {
          ownerId,
          coordinationRunId,
          coordinationRequestId,
          signal,
          onEvent: publish,
        }),
      )
      if (!initial.run.delegation) return this.resultEnvelope(initial)
      const delegation = initial.run.delegation
      publish({
        type: BackendEventType.DELEGATED,
        delegation: {
          id: delegation.id,
          sessionId: delegation.sessionId,
          title: delegation.title,
          directory: delegation.directory,
        },
      })
      const target = await delegation.promise
      publish({
        type: BackendEventType.DELEGATION_COMPLETED,
        delegation: {
          id: target.id,
          sessionId: target.sessionId,
          title: target.title,
          directory: target.directory,
        },
      })
      const final = await this.serialize(
        `coordinator:${key}`,
        () => this.coordinatorTurnWithRecovery(
          this.delegationResultPrompt(target, workObjective),
          {
            ownerId,
            coordinationRunId,
            coordinationRequestId,
            signal,
            onEvent: publish,
          },
        ),
      )
      return this.resultEnvelope(
        final,
        target,
        initial.result.contentBlocks || [],
      )
    } finally {
      if (runId) this.coordinationRuns.delete(runId)
    }
  }

  async canRecoverDelegatedWork(task) {
    const ownerId = clean(task?.ownerId)
    const runId = clean(task?.delegation?.id)
    const sessionId = clean(task?.delegation?.sessionId)
    if (!ownerId || !runId || !sessionId || !this.nativeDelegationAdapter) {
      return false
    }
    const remembered = this.registry.getProject(
      projectSessionKey(ownerId, this.protocol, sessionId),
      {
        ownerId,
        // This owner-scoped key is sufficient to migrate the short-lived
        // intermediate format that omitted the duplicate ownerId field.
        allowUnowned: true,
      },
    )
    if (!remembered || clean(remembered.sessionId) !== sessionId) return false
    const directory = clean(task?.delegation?.directory)
    if (directory && clean(remembered.cwd) !== directory) return false

    if (typeof this.nativeDelegationAdapter.canRecover !== 'function') {
      return false
    }
    const controller = new AbortController()
    let timer
    try {
      const timeout = new Promise(resolvePromise => {
        timer = setTimeout(() => {
          controller.abort(new Error('Delegation recovery probe timed out'))
          resolvePromise(false)
        }, this.recoveryProbeTimeoutMs)
        timer.unref?.()
      })
      const validation = Promise.race([
        Promise.resolve().then(() => (
          this.nativeDelegationAdapter.canRecover({
            runId,
            sessionId,
            ownerId,
            signal: controller.signal,
          })
        )),
        new Promise((_, reject) => {
          if (controller.signal.aborted) {
            reject(controller.signal.reason)
            return
          }
          controller.signal.addEventListener('abort', () => {
            reject(controller.signal.reason)
          }, { once: true })
        }),
      ]).catch(() => false)
      return await Promise.race([validation, timeout]) === true
    } catch {
      return false
    } finally {
      clearTimeout(timer)
    }
  }

  canRecoverDelegatedWorkSync() {
    // This adapter requires live native proof. The retained synchronous JS
    // recovery path cannot obtain it and must fail closed without starting a
    // detached probe.
    return false
  }

  async recoverDelegatedWork(task, {
    signal,
    onEvent,
  } = {}) {
    if (!await this.canRecoverDelegatedWork(task)) {
      throw new AgentError(`${this.label} 无法恢复这项第三层任务`, {
        protocol: this.protocol,
      })
    }
    const ownerId = clean(task.ownerId)
    const coordinationRunId = clean(task.id)
    const publish = event => this.publishWorkEvent(event, {
      taskId: coordinationRunId,
      ownerId,
      onEvent,
    })
    const key = coordinatorKey(ownerId, this.protocol)
    const session = await this.ensureCoordinatorSession(ownerId)
    const run = {
      ownerId,
      coordinationRunId,
      coordinationRequestId: coordinationRunId,
      onEvent: publish,
      sessionId: session.sessionId,
      nativeToolCalls: new Map(),
      toolCalls: new Map(),
      initialPromptDone: true,
      delegation: null,
    }
    const saved = task.delegation
    const delegation = this.createNativeDelegation(run, {
      delegationId: clean(saved.id),
      sessionId: clean(saved.sessionId),
      directory: clean(saved.directory || this.directory),
      title: bounded(saved.title || task.objective, 160)
        || this.profile.defaultDelegationTitle
        || `${this.label} 项目任务`,
    })
    if (coordinationRunId) this.coordinationRuns.set(coordinationRunId, run)
    signal?.addEventListener('abort', () => {
      delegation.controller.abort(
        signal.reason || new Error('用户已取消这项项目任务'),
      )
    }, { once: true })
    publish({
      type: BackendEventType.DELEGATED,
      delegation: {
        id: delegation.id,
        sessionId: delegation.sessionId,
        title: delegation.title,
        directory: delegation.directory,
        presentation: saved.presentation || null,
      },
    })
    try {
      const target = await delegation.promise
      publish({
        type: BackendEventType.DELEGATION_COMPLETED,
        delegation: {
          id: target.id,
          sessionId: target.sessionId,
          title: target.title,
          directory: target.directory,
        },
      })
      const final = await this.serialize(
        `coordinator:${key}`,
        () => this.coordinatorTurn(
          this.delegationResultPrompt(target, task.objective),
          {
            ownerId,
            coordinationRunId,
            coordinationRequestId: coordinationRunId,
            signal,
            onEvent: publish,
          },
        ),
      )
      return this.resultEnvelope(final, target)
    } finally {
      this.coordinationRuns.delete(coordinationRunId)
    }
  }

  async cancelWork(taskId, { ownerId } = {}) {
    const id = clean(taskId)
    const owner = clean(ownerId)
    const run = this.coordinationRuns.get(id)
    const record = run?.delegation
    const activeOwner = run?.ownerId || this.workOwners.get(id)
    if (!activeOwner || activeOwner !== owner) {
      throw new AgentError(`没有找到可取消的 ${this.label} 任务`, {
        protocol: this.protocol,
      })
    }
    if (record) {
      await this.cancelDelegation(
        { delegation_id: record.id },
        run.ownerId,
      )
    }

    // ACP cancellation interrupts execution but does not let a client rewrite
    // the backend-owned Session history. Record one terminal boundary only
    // when the request actually reached that persistent Session; it will be
    // projected into the next coordinator turn and acknowledged on success.
    if (run?.sessionId) {
      this.registry.appendReconciliation(
        coordinatorKey(owner, this.protocol),
        {
          kind: 'coordination_request_terminated',
          request_id: run.coordinationRequestId || clean(taskId),
          outcome: 'cancelled',
          ...(record
            ? {
                delegation_id: record.id,
                target_session_id: record.sessionId,
              }
            : {}),
          confirmed_at: new Date().toISOString(),
        },
      )
    }
    return {
      route: 'adapter',
      layer: record ? 'delegated' : 'coordinator',
      ...(record
        ? {
            delegationId: record.id,
            sessionId: record.sessionId,
          }
        : {}),
    }
  }

  async cancel(taskId, options = {}) {
    const id = clean(taskId)
    const controller = this.workControllers.get(id)
    const run = this.coordinationRuns.get(id)
    const activeOwner = run?.ownerId || this.workOwners.get(id)
    if (!controller && !run) {
      return { taskId: id, state: 'not_found' }
    }
    if (!activeOwner || activeOwner !== clean(options.ownerId)) {
      throw new AgentError(`没有找到可取消的 ${this.label} 任务`, {
        protocol: this.protocol,
      })
    }
    await this.cancelWork(id, options)
    controller?.abort(new AgentError('用户已取消这项工作', {
      status: 499,
      protocol: this.protocol,
    }))
    return { taskId: id, state: 'cancelled' }
  }

  async respondAuthorization(
    taskId,
    authorizationId,
    decision,
    { ownerId } = {},
  ) {
    const pending = this.pendingPermissions.get(clean(authorizationId))
    if (pending && clean(taskId) && pending.taskId !== clean(taskId)) {
      throw new AgentError('权限请求不属于这项工作', {
        status: 404,
        protocol: this.protocol,
      })
    }
    return this.resolveAuthorization(authorizationId, decision, { ownerId })
  }

  async queryDelegatedWork(taskId, _question, { ownerId } = {}) {
    const run = this.coordinationRuns.get(clean(taskId))
    const record = run?.delegation
    if (!record || record.ownerId !== clean(ownerId)) {
      throw new AgentError(`没有找到对应的 ${this.label} 项目任务`, {
        protocol: this.protocol,
      })
    }
    const status = this.statusForDelegation(
      { delegation_id: record.id },
      record.ownerId,
    )
    const latest = status.recent_updates?.at(-1)
    const latestDetail = bounded(
      latest?.detail || latest?.tool || latest?.kind,
      240,
    )
    const speech = status.status === 'completed'
      ? `这项工作已经完成：${clean(status.result)}`
      : status.status === 'failed'
        ? `这项工作已经失败：${clean(status.error)}`
        : status.status === 'cancelled'
          ? '这项工作已经取消。'
          : latestDetail
            ? `这项工作仍在执行中，最近进展：${latestDetail}`
            : '这项工作仍在执行中，当前没有新的详细进展。'
    return {
      content: JSON.stringify({
        task_id: clean(record.taskId) || clean(taskId),
        state: 'completed',
        mode: 'respond',
        presentation: { speech, inline: null },
      }),
      protocol: this.protocol,
      metadata: {
        statusSource: 'gateway',
        recentUpdates: status.recent_updates || [],
        delegation: {
          id: record.id,
          sessionId: record.sessionId,
          title: record.title,
          directory: record.directory,
        },
      },
    }
  }

  async uiUrl(ownerId) {
    if (!this.profile.uiUrl) return null
    const key = coordinatorKey(ownerId, this.protocol)
    const session = this.coordinatorSessions.get(key) || this.registry.get(key, {
      ownerId,
      allowUnowned: true,
    })
    return this.profile.uiUrl({
      baseUrl: this.baseUrl,
      sessionId: session?.sessionId || null,
      ownerId,
    })
  }

  async close() {
    for (const controller of this.workControllers.values()) {
      controller.abort(new AgentError('后台 Agent 已关闭', {
        status: 503,
        protocol: this.protocol,
      }))
    }
    this.workControllers.clear()
    this.workOwners.clear()
    for (const run of this.coordinationRuns.values()) {
      run.delegation?.controller.abort(
        new Error(`${this.label} backend is shutting down`),
      )
    }
    this.permissionBroker.cancelAll()
    await Promise.allSettled(
      [...this.coordinatorToolRegistrationPromises.values()],
    )
    const registrations = [...this.coordinatorToolRegistrations.values()]
    this.coordinatorToolRegistrations.clear()
    await Promise.allSettled(registrations.map(registration => (
      Promise.resolve().then(() => registration.release())
    )))
    await Promise.allSettled([
      this.sessionToolServer.close(),
      this.client.close(),
    ])
    await this.builtinMcpLifecycle.close()
    this.workEventListeners.clear()
    this.runtimeState.stopped()
  }
}
