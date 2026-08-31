#!/usr/bin/env node

import {
  existsSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const SCRIPT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const WORKSPACES = ['server', 'web', 'tui', 'desktop', 'cli']
const MANIFESTS = [
  'package.json',
  ...WORKSPACES.map(workspace => `${workspace}/package.json`),
]
const VERSION_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(?:0|[1-9]\d*|[a-zA-Z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|[A-Za-z-][0-9A-Za-z-]*))*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$/
const PEP440_PRE_RELEASE = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(alpha|beta|rc)\.([0-9]\d*))?(?:\+([0-9A-Za-z.-]+))?$/i
const PRE_RELEASE_TAGS = { alpha: 'a', beta: 'b', rc: 'rc' }

export function isValidVersion(version) {
  return VERSION_PATTERN.test(String(version))
}

export function toPep440(version) {
  const match = String(version).match(PEP440_PRE_RELEASE)
  if (!match) {
    throw new Error(`无法转换为 PEP 440 版本：${version}`)
  }
  const [, major, minor, patch, label, number, local] = match
  const core = `${major}.${minor}.${patch}`
  const tagged = label ? `${core}${PRE_RELEASE_TAGS[label.toLowerCase()]}${number}` : core
  return local ? `${tagged}+${local}` : tagged
}

export function readDeclaredPythonVersion(root) {
  const pyprojectPath = resolve(root, 'pyproject.toml')
  const initPath = resolve(root, 'src/qwen_audio_agent/__init__.py')
  if (!existsSync(pyprojectPath) || !existsSync(initPath)) {
    throw new Error('缺少 pyproject.toml 或 src/qwen_audio_agent/__init__.py')
  }
  const pyproject = readFileSync(pyprojectPath, 'utf8').match(/^version = "([^"]+)"/m)
  const module = readFileSync(initPath, 'utf8').match(/^__version__ = "([^"]+)"/m)
  if (!pyproject) throw new Error('pyproject.toml 缺少 version')
  if (!module) throw new Error('src/qwen_audio_agent/__init__.py 缺少 __version__')
  return { pyproject: pyproject[1], module: module[1] }
}

export function assertPythonVersionsAligned(root, npmVersion) {
  const { pyproject, module } = readDeclaredPythonVersion(root)
  if (pyproject !== module) {
    throw new Error(
      `pyproject.toml 版本 ${pyproject} 与 __version__ ${module} 不一致`,
    )
  }
  const major = Number(String(npmVersion).split('.', 1)[0])
  if (Number.isFinite(major) && major >= 2) {
    const expected = toPep440(npmVersion)
    if (pyproject !== expected) {
      throw new Error(
        `Python 版本 ${pyproject} 必须对应 npm ${npmVersion}（PEP 440：${expected}）`,
      )
    }
  }
  const uvLockPath = resolve(root, 'uv.lock')
  if (!existsSync(uvLockPath)) return
  const locked = readFileSync(uvLockPath, 'utf8').match(
    /name = "qwen-audio-agent-python"\nversion = "([^"]+)"/,
  )
  if (!locked) {
    throw new Error('uv.lock 缺少 qwen-audio-agent-python 版本记录')
  }
  if (locked[1] !== pyproject) {
    throw new Error(
      `uv.lock 版本 ${locked[1]} 与 pyproject.toml ${pyproject} 不一致`,
    )
  }
}

export function nextVersion(current, requested) {
  if (isValidVersion(requested)) return requested
  if (!['patch', 'minor', 'major'].includes(requested)) {
    throw new Error(
      '请指定完整版本号或 patch、minor、major，例如：'
      + 'npm run version:set -- 0.5.1',
    )
  }
  if (!isValidVersion(current)) {
    throw new Error(`当前版本号无效：${current}`)
  }
  const [major, minor, patch] = current
    .split('-', 1)[0]
    .split('.')
    .map(Number)
  if (requested === 'major') return `${major + 1}.0.0`
  if (requested === 'minor') return `${major}.${minor + 1}.0`
  return `${major}.${minor}.${patch + 1}`
}

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'))
}

function serializeJson(value) {
  return `${JSON.stringify(value, null, 2)}\n`
}

function writeAtomic(path, content) {
  const temporaryPath = `${path}.${process.pid}.tmp`
  try {
    writeFileSync(temporaryPath, content)
    renameSync(temporaryPath, path)
  } catch (error) {
    try {
      unlinkSync(temporaryPath)
    } catch {
      // The temporary file may not have been created.
    }
    throw error
  }
}

export function updateVersions(root, requested) {
  const rootManifest = readJson(resolve(root, 'package.json'))
  const version = nextVersion(rootManifest.version, requested)
  const updates = MANIFESTS.map(relativePath => {
    const path = resolve(root, relativePath)
    const manifest = readJson(path)
    if (!manifest.name) {
      throw new Error(`${relativePath} 缺少 package name`)
    }
    manifest.version = version
    return { path, content: serializeJson(manifest) }
  })

  const lockPath = resolve(root, 'package-lock.json')
  const lock = readJson(lockPath)
  if (!lock.packages?.['']) {
    throw new Error('package-lock.json 缺少根包记录')
  }
  lock.version = version
  lock.packages[''].version = version
  for (const workspace of WORKSPACES) {
    if (!lock.packages[workspace]) {
      throw new Error(`package-lock.json 缺少 workspace：${workspace}`)
    }
    lock.packages[workspace].version = version
  }
  updates.push({ path: lockPath, content: serializeJson(lock) })

  const pep440 = toPep440(version)
  const pyprojectPath = resolve(root, 'pyproject.toml')
  if (existsSync(pyprojectPath)) {
    const pyproject = readFileSync(pyprojectPath, 'utf8')
    if (!/^version = "/m.test(pyproject)) {
      throw new Error('pyproject.toml 缺少 version')
    }
    updates.push({
      path: pyprojectPath,
      content: pyproject.replace(/^version = "[^"]+"/m, `version = "${pep440}"`),
    })
  }
  const initPath = resolve(root, 'src/qwen_audio_agent/__init__.py')
  if (existsSync(initPath)) {
    const initSource = readFileSync(initPath, 'utf8')
    if (!/^__version__ = "/m.test(initSource)) {
      throw new Error('src/qwen_audio_agent/__init__.py 缺少 __version__')
    }
    updates.push({
      path: initPath,
      content: initSource.replace(/^__version__ = "[^"]+"/m, `__version__ = "${pep440}"`),
    })
  }
  const uvLockPath = resolve(root, 'uv.lock')
  if (existsSync(uvLockPath)) {
    const uvLock = readFileSync(uvLockPath, 'utf8')
    const locked = uvLock.replace(
      /(name = "qwen-audio-agent-python"\nversion = ")[^"]+(")/,
      `$1${pep440}$2`,
    )
    if (locked === uvLock && !uvLock.includes(`version = "${pep440}"`)) {
      throw new Error('uv.lock 缺少 qwen-audio-agent-python 版本记录')
    }
    updates.push({ path: uvLockPath, content: locked })
  }

  for (const relativePath of ['README.md', 'README_ZH.md']) {
    const path = resolve(root, relativePath)
    if (!existsSync(path)) continue
    const content = readFileSync(path, 'utf8')
    const updated = content.replace(
      /img\.shields\.io\/badge\/npm-v[^-)]+-orange/g,
      `img.shields.io/badge/npm-v${version}-orange`,
    )
    updates.push({ path, content: updated })
  }

  for (const update of updates) writeAtomic(update.path, update.content)
  return {
    previousVersion: rootManifest.version,
    version,
    changed: rootManifest.version !== version,
  }
}

const isMain = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href

if (isMain) {
  try {
    const result = updateVersions(SCRIPT_ROOT, process.argv[2])
    if (result.changed) {
      process.stdout.write(
        `版本已更新：${result.previousVersion} → ${result.version}\n`
        + `请更新 CHANGELOG.md，然后运行 npm run release:check。\n`,
      )
    } else {
      process.stdout.write(`版本已经是 ${result.version}。\n`)
    }
  } catch (error) {
    process.stderr.write(`版本更新失败：${error.message}\n`)
    process.exitCode = 1
  }
}
