import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const root = new URL('../../', import.meta.url)

test('builds and uploads both install and automatic-update macOS artifacts', () => {
  const builder = readFileSync(new URL('desktop/electron-builder.yml', root), 'utf8')
  const workflow = readFileSync(new URL('.github/workflows/release.yml', root), 'utf8')
  const manifest = JSON.parse(readFileSync(new URL('package.json', root), 'utf8'))

  assert.match(builder, /target:\s*\n\s*- dmg\s*\n\s*- zip/)
  assert.match(manifest.scripts['desktop:build'], /--mac dmg zip/)
  assert.match(workflow, /dist\/desktop\/\*\.dmg/)
  assert.match(workflow, /dist\/desktop\/\*\.zip/)
})

test('ships the locked Python Gateway project outside app.asar', () => {
  const builder = readFileSync(new URL('desktop/electron-builder.yml', root), 'utf8')
  const resources = builder.split('extraResources:')[1] || ''

  for (const entry of [
    'from: src/qwen_audio_agent',
    'to: runtime/src/qwen_audio_agent',
    'from: pyproject.toml',
    'to: runtime/pyproject.toml',
    'from: uv.lock',
    'to: runtime/uv.lock',
    'from: README.md',
    'to: runtime/README.md',
  ]) {
    assert.equal(resources.includes(entry), true, `missing packaged resource: ${entry}`)
  }
  assert.equal(resources.includes('!**/__pycache__/**'), true)
  assert.equal(resources.includes('!**/*.pyc'), true)
})
