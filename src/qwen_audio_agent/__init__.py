"""Python core for qwen-audio-agent.

Independent Apache-2.0 derivative of https://github.com/QwenAudio/qwen-audio-agent.
Not an official QwenAudio release.

The public v2 baseline is a voice Gateway product: LangGraph owns two bounded
state machines (FrontendTurnGraph and WorkGraph). Audio, models, and ACP
backends stay outside the graphs. Node and Electron hosts remain; ACP adapters
stay in JavaScript. This is not a LangGraph Platform app or a reusable graph
library.
"""

__version__ = "2.0.0a0"
