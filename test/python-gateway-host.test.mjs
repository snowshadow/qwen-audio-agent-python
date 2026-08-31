import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { DEFAULT_GATEWAY_ENTRY } from '../shared/gateway-process.mjs'
import {
  GATEWAY_READY_MESSAGE,
  parseGatewayIpcLine,
  pythonGatewayEnvironment,
  resolvePythonLaunch,
} from '../shared/python-gateway-host.mjs'

test('the existing host API points at the thin Python bridge', () => {
  assert.equal(
    DEFAULT_GATEWAY_ENTRY.endsWith('/server/src/python-gateway-entry.mjs'),
    true,
  )
})

test('the thin entry loads config.env before enforcing the setup gate', () => {
  const entry = readFileSync(
    new URL('../server/src/python-gateway-entry.mjs', import.meta.url),
    'utf8',
  )
  const loadAt = entry.indexOf('loadRuntimeEnvironment({ root })')
  const gateAt = entry.indexOf('assertGatewaySetup()')

  assert.notEqual(loadAt, -1)
  assert.equal(loadAt < gateAt, true)
})

test('resolves an explicit Python command without shell parsing', () => {
  const launch = resolvePythonLaunch({
    root: '/runtime',
    env: {
      PYTHONPATH: '/shared/python',
      QWEN_AUDIO_AGENT_PYTHON: '/opt/python/bin/python',
      QWEN_AUDIO_AGENT_PYTHON_ARGS_JSON: '["--log-level","debug"]',
    },
  })

  assert.equal(launch.executable, '/opt/python/bin/python')
  assert.deepEqual(launch.args, [
    '-m',
    'qwen_audio_agent.cli',
    '--log-level',
    'debug',
  ])
  assert.equal(launch.cwd, resolve('/runtime'))
  assert.equal(
    launch.env.PYTHONPATH,
    `${resolve('/runtime/src')}:/shared/python`,
  )
})

test('a packaged tree without a venv uses the locked uv project', () => {
  const launch = resolvePythonLaunch({
    root: '/definitely/missing/qwenaudio-runtime',
    env: { QWEN_AUDIO_AGENT_UV: '/opt/bin/uv' },
  })

  assert.equal(launch.executable, '/opt/bin/uv')
  assert.deepEqual(launch.args.slice(0, 7), [
    'run',
    '--project',
    resolve('/definitely/missing/qwenaudio-runtime'),
    '--frozen',
    '--no-dev',
    'python',
    '-m',
  ])
  assert.equal(
    launch.env.PYTHONPATH,
    resolve('/definitely/missing/qwenaudio-runtime/src'),
  )
})

test('locates the packaged runtime root from the desktop environment', () => {
  const runtimeRoot = '/Applications/Qwen.app/Contents/Resources/runtime'
  const launch = resolvePythonLaunch({
    env: {
      QWAUDIO_CONFIG_DIR: '/Users/test/Library/Application Support/Qwen Audio Agent',
      QWEN_AUDIO_AGENT_RUNTIME_ROOT: runtimeRoot,
      QWEN_AUDIO_AGENT_UV: '/opt/bin/uv',
    },
  })

  assert.equal(launch.cwd, resolve(runtimeRoot))
  assert.equal(launch.args[2], resolve(runtimeRoot))
  assert.equal(launch.env.PYTHONPATH, resolve(runtimeRoot, 'src'))
  assert.equal(
    launch.env.UV_PROJECT_ENVIRONMENT,
    resolve(
      '/Users/test/Library/Application Support/Qwen Audio Agent',
      'state/python-environment',
    ),
  )
})

test('passes the current Node-compatible executable through to Python', () => {
  const environment = pythonGatewayEnvironment({
    env: { QWEN_AUDIO_AGENT_SOURCE_ROOT: '/resources/app.asar' },
    launchEnv: { PYTHONPATH: '/resources/runtime/src' },
    nodeExecutable: '/Applications/Qwen.app/Contents/MacOS/Qwen Audio Agent',
  })

  assert.equal(
    environment.QWEN_AUDIO_AGENT_NODE,
    '/Applications/Qwen.app/Contents/MacOS/Qwen Audio Agent',
  )
  assert.equal(environment.QWEN_AUDIO_AGENT_SOURCE_ROOT, '/resources/app.asar')
  assert.equal(environment.PYTHONPATH, '/resources/runtime/src')
})

test('rejects shell-shaped Python argument configuration', () => {
  assert.throws(
    () => resolvePythonLaunch({
      root: '/runtime',
      env: { QWEN_AUDIO_AGENT_PYTHON_ARGS_JSON: '"--unsafe; command"' },
    }),
    /JSON string array/,
  )
})

test('parses only structured child IPC lines', () => {
  assert.deepEqual(
    parseGatewayIpcLine(JSON.stringify({
      type: GATEWAY_READY_MESSAGE,
      origin: 'http://127.0.0.1:3101',
    })),
    {
      type: GATEWAY_READY_MESSAGE,
      origin: 'http://127.0.0.1:3101',
    },
  )
  assert.equal(parseGatewayIpcLine('ordinary log output'), null)
  assert.equal(
    parseGatewayIpcLine(JSON.stringify({ type: GATEWAY_READY_MESSAGE })),
    null,
  )
})
