import { logger } from '../core/logger.mjs'
import { VersionedJsonStore } from '../core/versioned-json-store.mjs'

const VERSION = 1
const MAX_RECONCILIATIONS = 20

function clean(value) {
  return String(value || '').trim()
}

function reconciliationKey(fact = {}) {
  return [fact.kind, fact.request_id, fact.outcome].map(String).join('\u0000')
}

export class AcpSessionRegistry {
  constructor({
    filePath = null,
    onWarning = warning => logger.warn(
      'acp.session_index_persistence_warning',
      { warning },
    ),
  } = {}) {
    this.filePath = filePath
    this.store = new VersionedJsonStore({
      filePath,
      version: VERSION,
      label: 'ACP Session 索引',
      onWarning,
    })
    this.loaded = false
    this.coordinators = {}
    this.projects = {}
    this.reconciliations = {}
  }

  load() {
    if (this.loaded) return
    this.loaded = true
    const parsed = this.store.load({
      fallback: () => ({ coordinators: {}, projects: {}, reconciliations: {} }),
      validate: value => Boolean(
        value.coordinators
        && typeof value.coordinators === 'object'
        && !Array.isArray(value.coordinators)
        && (value.projects === undefined || (
          value.projects
          && typeof value.projects === 'object'
          && !Array.isArray(value.projects)
        ))
        && (value.reconciliations === undefined || (
          value.reconciliations
          && typeof value.reconciliations === 'object'
          && !Array.isArray(value.reconciliations)
        )),
      ),
    })
    this.coordinators = parsed.coordinators
    this.projects = parsed.projects || {}
    this.reconciliations = parsed.reconciliations || {}
  }

  get(key, { ownerId, allowUnowned = false } = {}) {
    this.load()
    const value = this.coordinators[String(key)]
    if (!value || typeof value !== 'object') return null
    const expectedOwner = clean(ownerId)
    const actualOwner = clean(value.ownerId)
    if (
      expectedOwner
      && (actualOwner ? actualOwner !== expectedOwner : !allowUnowned)
    ) return null
    return { ...value }
  }

  set(key, session, { ownerId } = {}) {
    this.load()
    const resolvedOwner = clean(ownerId || session.ownerId)
    this.coordinators[String(key)] = {
      sessionId: String(session.sessionId),
      cwd: String(session.cwd),
      ...(resolvedOwner ? { ownerId: resolvedOwner } : {}),
      ...(Number.isInteger(session.contractVersion)
        ? { contractVersion: session.contractVersion }
        : {}),
      updatedAt: Date.now(),
    }
    this.save()
  }

  delete(key) {
    this.load()
    delete this.coordinators[String(key)]
    delete this.reconciliations[String(key)]
    this.save()
  }

  reconciliationsFor(key) {
    this.load()
    const values = this.reconciliations[String(key)]
    return Array.isArray(values) ? values.map(value => ({ ...value })) : []
  }

  appendReconciliation(key, fact) {
    this.load()
    const storageKey = String(key)
    const values = this.reconciliationsFor(storageKey)
    const identity = reconciliationKey(fact)
    const next = values.filter(value => reconciliationKey(value) !== identity)
    next.push({ ...fact })
    this.reconciliations[storageKey] = next.slice(-MAX_RECONCILIATIONS)
    this.save()
  }

  acknowledgeReconciliations(key, facts = []) {
    this.load()
    const storageKey = String(key)
    const acknowledged = new Set(facts.map(reconciliationKey))
    if (!acknowledged.size) return
    const remaining = this.reconciliationsFor(storageKey).filter(
      value => !acknowledged.has(reconciliationKey(value)),
    )
    if (remaining.length) this.reconciliations[storageKey] = remaining
    else delete this.reconciliations[storageKey]
    this.save()
  }

  getProject(key, { ownerId, allowUnowned = false } = {}) {
    this.load()
    const value = this.projects[String(key)]
    if (!value || typeof value !== 'object') return null
    const expectedOwner = clean(ownerId)
    const actualOwner = clean(value.ownerId)
    if (
      expectedOwner
      && (actualOwner ? actualOwner !== expectedOwner : !allowUnowned)
    ) return null
    return { ...value }
  }

  setProject(key, session, { ownerId } = {}) {
    this.setProjects([[key, session]], { ownerId })
  }

  setProjects(entries, { ownerId } = {}) {
    this.load()
    let changed = false
    for (const [key, session] of entries) {
      const sessionId = String(session.sessionId || '')
      const cwd = String(session.cwd || '')
      if (!sessionId || !cwd) continue
      const resolvedOwner = clean(ownerId || session.ownerId)
      this.projects[String(key)] = {
        sessionId,
        cwd,
        title: String(session.title || ''),
        ...(resolvedOwner ? { ownerId: resolvedOwner } : {}),
        updatedAt: Date.now(),
      }
      changed = true
    }
    if (changed) this.save()
  }

  deleteProject(key) {
    this.load()
    if (!Object.hasOwn(this.projects, String(key))) return false
    delete this.projects[String(key)]
    this.save()
    return true
  }

  save() {
    this.store.save({
      coordinators: this.coordinators,
      projects: this.projects,
      reconciliations: this.reconciliations,
    })
  }

  health() {
    return this.store.health()
  }
}
