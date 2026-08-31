import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

export const PYTHON_GATEWAY_MODULE = 'qwen_audio_agent.cli'
export const GATEWAY_READY_MESSAGE = 'qwen-audio-agent:gateway-ready'

export function pythonGatewayEnvironment({
  env = process.env,
  launchEnv = {},
  nodeExecutable = process.execPath,
} = {}) {
  const childEnv = { ...env, ...launchEnv }
  if (!String(childEnv.QWEN_AUDIO_AGENT_NODE || '').trim()) {
    childEnv.QWEN_AUDIO_AGENT_NODE = nodeExecutable
  }
  return childEnv
}

export function resolvePythonLaunch({ root, env = process.env, platform = process.platform } = {}) {
  const runtimeRoot = resolve(
    root || env.QWEN_AUDIO_AGENT_RUNTIME_ROOT || process.cwd(),
  )
  const explicit = String(env.QWEN_AUDIO_AGENT_PYTHON || '').trim()
  const managedPython = platform === 'win32'
    ? resolve(runtimeRoot, '.venv/Scripts/python.exe')
    : resolve(runtimeRoot, '.venv/bin/python')
  const separator = platform === 'win32' ? ';' : ':'
  const sourceDirectory = resolve(runtimeRoot, 'src')
  const inheritedPythonPath = String(env.PYTHONPATH || '')
    .split(separator)
    .filter(Boolean)
  const pythonPath = [...new Set([sourceDirectory, ...inheritedPythonPath])]
    .join(separator)
  const childEnv = { PYTHONPATH: pythonPath }
  const configuredUvEnvironment = String(env.UV_PROJECT_ENVIRONMENT || '').trim()
  const configDirectory = String(env.QWAUDIO_CONFIG_DIR || '').trim()
  if (configuredUvEnvironment) {
    childEnv.UV_PROJECT_ENVIRONMENT = configuredUvEnvironment
  } else if (configDirectory) {
    // Packaged application resources are read-only. Keep uv's managed
    // environment with the rest of this Desktop instance's runtime state.
    childEnv.UV_PROJECT_ENVIRONMENT = resolve(
      configDirectory,
      'state/python-environment',
    )
  }

  let extraArgs = []
  const encodedArgs = String(env.QWEN_AUDIO_AGENT_PYTHON_ARGS_JSON || '').trim()
  if (encodedArgs) {
    const parsed = JSON.parse(encodedArgs)
    if (!Array.isArray(parsed) || parsed.some(value => typeof value !== 'string')) {
      throw new TypeError('QWEN_AUDIO_AGENT_PYTHON_ARGS_JSON must be a JSON string array')
    }
    extraArgs = parsed
  }
  if (explicit || existsSync(managedPython)) {
    return {
      executable: explicit || managedPython,
      args: ['-m', PYTHON_GATEWAY_MODULE, ...extraArgs],
      cwd: runtimeRoot,
      env: childEnv,
    }
  }
  const uv = String(env.QWEN_AUDIO_AGENT_UV || 'uv').trim()
  return {
    executable: uv,
    args: [
      'run', '--project', runtimeRoot, '--frozen', '--no-dev',
      'python', '-m', PYTHON_GATEWAY_MODULE, ...extraArgs,
    ],
    cwd: runtimeRoot,
    env: childEnv,
  }
}

export function parseGatewayIpcLine(line) {
  const value = String(line || '').trim()
  if (!value.startsWith('{')) return null
  try {
    const message = JSON.parse(value)
    if (!message || typeof message !== 'object' || typeof message.type !== 'string') {
      return null
    }
    if (
      message.type === GATEWAY_READY_MESSAGE
      && typeof message.origin !== 'string'
    ) return null
    return message
  } catch {
    return null
  }
}
