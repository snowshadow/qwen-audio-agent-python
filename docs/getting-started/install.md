# Installation

Requires Node.js 22.22.2+ or 24.15.0+, npm 10+, Python 3.12–3.14, and
[uv](https://docs.astral.sh/uv/). The first Gateway launch prepares the locked
Python runtime with `uv run --frozen --no-dev`. When using the default DashScope
real-time voice frontend, a DashScope API Key is also required.
The repository provides `.nvmrc` and `.node-version`; when using nvm, you can simply run `nvm use`.

This tree is an independent Apache-2.0 derivative of
[QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent),
not an official QwenAudio release. The commands below install **this**
repository. `npm install -g qwen-audio-agent` still installs the upstream
project.

## Install from GitHub

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

## Install from Source

```bash
git clone https://github.com/snowshadow/qwen-audio-agent-python.git
cd qwen-audio-agent-python
uv sync --frozen
npm install
npm run install:global
```

## Upgrade

Upgrade to the latest code from this repository:

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

After upgrading, if the Gateway is running as a background service, run `qwenaudio gateway restart` for the new version to take effect.

## Verify Installation

View the exact location of the configuration file and confirm the installation is ready:

```bash
qwenaudio config
```

After configuring the backend agent, you can run a read-only check to confirm whether the backend executable, ACP integration, and adapter are ready:

```bash
qwenaudio setup
```

## Configuration File Location

The CLI uses `~/.config/qwaudio/config.env`; the desktop version uses the system's standard application data directory
(`~/Library/Application Support/Qwen Audio Agent` on macOS). The two have separate Gateways,
locks, logs, and settings, and can run simultaneously. Set `QWAUDIO_CONFIG_DIR` or
`XDG_CONFIG_HOME` to change the configuration directory. See [Configuration](../configuration.md) for details.

## Obtain a DashScope API Key

Alibaba Cloud Model Studio (Bailian) provides a
[free trial quota](https://help.aliyun.com/zh/model-studio/new-free-quota) for Qwen Audio 3.0 Realtime. After creating an API Key,
you can start using qwen-audio-agent for free.

1. Open the [API Key page](https://bailian.console.aliyun.com/?tab=model#/api-key) in the Bailian console,
   log in to your account, and click **Create API Key**.
2. Copy the generated Key and fill it into `config.env` later. Do not publicly share or commit your API Key.

For detailed instructions, see the [official Bailian documentation](https://help.aliyun.com/zh/model-studio/get-api-key).
