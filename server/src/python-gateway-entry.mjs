import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createInterface } from 'node:readline'
import {
  parseGatewayIpcLine,
  pythonGatewayEnvironment,
  resolvePythonLaunch,
} from '../../shared/python-gateway-host.mjs'
import { assertGatewaySetup } from '../../shared/gateway-setup.mjs'
import { loadRuntimeEnvironment } from '../../shared/runtime-environment.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
loadRuntimeEnvironment({ root })
try {
  assertGatewaySetup()
} catch (error) {
  console.error(error.message)
  process.exit(1)
}
const launch = resolvePythonLaunch({
  root: process.env.QWEN_AUDIO_AGENT_RUNTIME_ROOT || root,
})
const child = spawn(launch.executable, launch.args, {
  cwd: launch.cwd,
  env: pythonGatewayEnvironment({ launchEnv: launch.env }),
  stdio: ['ignore', 'pipe', 'inherit'],
})

let stopping = false
let forwardedSignal = null

createInterface({ input: child.stdout }).on('line', line => {
  const message = parseGatewayIpcLine(line)
  if (message) process.send?.(message)
  else process.stdout.write(`${line}\n`)
})

function stop(signal = 'SIGTERM') {
  if (stopping) return
  stopping = true
  forwardedSignal = signal
  child.kill(signal)
}

process.once('SIGINT', () => stop('SIGINT'))
process.once('SIGTERM', () => stop('SIGTERM'))
process.once('disconnect', () => stop('SIGTERM'))

child.once('error', error => {
  console.error(`Python Gateway failed to start: ${error.message}`)
  process.exitCode = 1
})

child.once('exit', (code, signal) => {
  if (signal && signal !== forwardedSignal) {
    process.exit(1)
    return
  }
  process.exit(code ?? (signal ? 1 : 0))
})
