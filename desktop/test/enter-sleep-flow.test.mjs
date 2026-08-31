import assert from 'node:assert/strict'
import test from 'node:test'
import { applyDesktopClientState } from '../../web/src/desktop-hide.js'
import { DesktopPresence } from '../src/desktop-presence.mjs'
import { GatewayServerEvent } from '../../shared/realtime-events.mjs'

test('a sleeping client state hides the desktop orb end to end', async () => {
  const window = {
    hidden: false,
    isDestroyed: () => false,
    hide() { this.hidden = true },
    webContents: { send() {} },
  }
  const presence = new DesktopPresence({
    getWindow: () => window,
    globalShortcut: {
      register: () => true,
      unregister() {},
      unregisterAll() {},
    },
  })
  const hidden = await applyDesktopClientState({
    type: GatewayServerEvent.CLIENT_STATE,
    state: 'sleeping',
  }, {
    desktop: true,
    bridge: {
      enterHide: async () => ({
        state: presence.hide('requested'),
      }),
    },
  })

  assert.equal(hidden, true)
  assert.equal(window.hidden, true)
  assert.equal(presence.state, 'hidden')
})
