import { createInterface } from 'node:readline'

function write(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`)
}

createInterface({ input: process.stdin }).on('line', line => {
  const request = JSON.parse(line)
  const { id, method, params = {} } = request
  if (method === 'start') return write({ id, result: { ok: true } })
  if (method === 'describe') {
    return write({ id, result: { protocol: 'fixture', label: 'Fixture' } })
  }
  if (method === 'health') {
    if (process.env.QWEN_AUDIO_TEST_HANG_HEALTH === '1') return undefined
    return write({ id, result: { ok: true } })
  }
  if (method === 'submit') {
    write({ event: {
      type: 'backend.activity',
      taskId: params.task.id,
      activity: { kind: 'status', message: 'fixture started' },
    } })
    const content = params.task.instruction === 'large'
      ? 'x'.repeat(100_000)
      : `done:${params.task.instruction}`
    return write({ id, result: { content } })
  }
  if (method === 'cancel') {
    return write({
      id,
      result: {
        taskId: params.taskId,
        state: 'cancelled',
        context: params.context,
      },
    })
  }
  if (method === 'status') return write({ id, result: { state: 'working' } })
  if (method === 'respondAuthorization') {
    return write({ id, result: { ok: true, context: params.context } })
  }
  if (method === 'canRecoverDelegatedWork') return write({ id, result: false })
  if (method === 'close') {
    write({ id, result: { ok: true } })
    return setImmediate(() => process.exit(0))
  }
  write({ id, error: { code: 'UNKNOWN', message: method } })
})
