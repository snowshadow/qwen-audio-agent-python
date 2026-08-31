#!/usr/bin/env python3
"""qwen-audio-agent 全屏工作台 TUI(Textual):两种终端风格中的"全屏"版。

与极简语音 TUI(src/index.mjs,npm run tui)的分工:
  - 极简版:纯语音、滚动输出、零依赖上手,适合快速对话
  - 全屏版(本文件):状态灯/任务面板/文字输入框/麦克风电平,适合边聊边干活

音频引擎按平台自动降级:
  1. macOS:CoreAudio Voice Processing 助手(native/macos-voice-io.swift,
     首次运行自动用 swiftc 编译,与极简版共享)——系统级回声消除,外放全双工
  2. 其他平台/无 swiftc:sounddevice 半双工(帧级门控防回声,回车/Ctrl+X 打断)
  3. 无音频设备:纯文字模式

用法(有麦克风的机器;仓库根目录可用 npm run tui:full):
  pip install aiohttp sounddevice textual
  python3 tui/fullscreen/app.py [--url http://127.0.0.1:3101] [--session voice-main]
                                [--headphones]  # 半双工下:关门控,语音打断
                                [--barge-in]    # 实验:外放能量突破打断(默认关)
"""

import argparse
import asyncio
import base64
import collections
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
from typing import ClassVar

import aiohttp

try:
    import audioop  # Python 3.13 起从标准库移除

    def pcm_rms(pcm):
        return audioop.rms(pcm, 2)
except ImportError:
    import array

    def pcm_rms(pcm):
        if not pcm:
            return 0
        samples = array.array("h", pcm[: len(pcm) // 2 * 2])
        return int((sum(s * s for s in samples) / len(samples)) ** 0.5)


IN_RATE, OUT_RATE = 16000, 24000
MIC_BLOCK = 1600  # 100ms 块:VAD 响应更快
MIC_GATE_TAIL_S = 0.35
BARGE_RMS_RATIO = 3.0  # 能量突破:超过回声底噪的倍数
BARGE_RMS_FLOOR = 1200  # 绝对下限,防止安静环境下底噪过低导致误触
BARGE_SUSTAIN_BLOCKS = 2  # 连续 200ms 判定为真人插话(一声"停"也够)
BARGE_PREROLL_BLOCKS = 4  # 突破时补发门控期缓存的最近 400ms,保住插话开头


def complete_transcript(streamed, final):
    """优先采用完整 final；若 final 是流式文本的短前缀，保留已收到的尾部。"""
    streamed = str(streamed or "").strip()
    final = str(final or "").strip()
    if not final or streamed.startswith(final):
        return streamed
    return final


def frontend_label(holder):
    holder = holder or {}
    return (
        holder.get("label")
        or {"desktop": "桌面端", "cli": "终端", "web": "WebUI"}.get(holder.get("type"))
        or "其他前端"
    )


class VoiceEngine:
    """采集/播放/门控/能量突破。线程回调 → loop.call_soon_threadsafe 回主循环。"""

    def __init__(self, loop, send_json, on_level, on_barge, mic_gate=True, barge_in=False):
        import sounddevice as sd

        self.loop = loop
        self.send_json = send_json
        self.on_level = on_level  # (mic_rms01, playing) -> None
        self.on_barge = on_barge  # () -> None 能量突破打断
        self.mic_gate = mic_gate
        self.barge_in = barge_in
        self.queue = collections.deque()  # (response_id, bytes)
        self.remaining = {}
        self.done_marks = set()
        self.started_sent = set()
        self.playing = False
        self.last_play_end = 0.0
        self.muted = False
        self._echo_baseline = 300.0  # 播放期麦克风 RMS 滚动底噪
        self._barge_run = 0
        self._preroll = collections.deque(maxlen=BARGE_PREROLL_BLOCKS)
        self.dbg = {"rms": 0, "thr": 0, "run": 0, "gated": False}

        self.instream = sd.RawInputStream(
            samplerate=IN_RATE,
            channels=1,
            dtype="int16",
            blocksize=MIC_BLOCK,
            callback=self._on_mic,
        )
        self.outstream = sd.RawOutputStream(
            samplerate=OUT_RATE, channels=1, dtype="int16", blocksize=2400, callback=self._on_spk
        )

    # ---- 采集 ----
    def _on_mic(self, data, frames, t, status):
        pcm = bytes(data)
        rms = pcm_rms(pcm)
        self.loop.call_soon_threadsafe(self.on_level, min(1.0, rms / 12000), self.playing)
        if self.muted:
            return
        gated = self.mic_gate and (
            self.playing or time.monotonic() - self.last_play_end < MIC_GATE_TAIL_S
        )
        self.dbg["gated"] = gated and self.barge_in
        if gated:
            if not self.barge_in:
                return  # 默认半双工:放音期间不采音,干净利落
            # 能量突破:播放期持续追踪回声底噪;明显盖过底噪的持续人声 → 打断并放行
            self._preroll.append(pcm)
            self._echo_baseline = 0.95 * self._echo_baseline + 0.05 * max(rms, 1)
            thr = max(self._echo_baseline * BARGE_RMS_RATIO, BARGE_RMS_FLOOR)
            self.dbg.update(rms=int(rms), thr=int(thr), run=self._barge_run)
            if rms > thr:
                self._barge_run += 1
                if self._barge_run >= BARGE_SUSTAIN_BLOCKS:
                    self._barge_run = 0
                    # 补发缓存的插话开头,ASR 才能听到完整的话
                    backlog = b"".join(self._preroll)
                    self._preroll.clear()
                    self.loop.call_soon_threadsafe(self.on_barge)
                    self.loop.call_soon_threadsafe(
                        self.send_json,
                        {"type": "audio.append", "audio": base64.b64encode(backlog).decode()},
                    )
            else:
                self._barge_run = 0
            return
        self._preroll.clear()
        self.loop.call_soon_threadsafe(
            self.send_json, {"type": "audio.append", "audio": base64.b64encode(pcm).decode()}
        )

    # ---- 播放(硬件按需拉取) ----
    def _on_spk(self, out, frames, t, status):
        need = len(out)
        got = bytearray()
        while self.queue and len(got) < need:
            rid, chunk = self.queue[0]
            if rid and rid not in self.started_sent:
                self.started_sent.add(rid)
                self.loop.call_soon_threadsafe(
                    self.send_json, {"type": "playback.started", "responseId": rid}
                )
            take = min(len(chunk), need - len(got))
            got += chunk[:take]
            if take == len(chunk):
                self.queue.popleft()
            else:
                self.queue[0] = (rid, chunk[take:])
            if rid:
                left = self.remaining.get(rid, 0) - take
                self.remaining[rid] = left
                if left <= 0 and rid in self.done_marks:
                    self._finish(rid)
        out[: len(got)] = bytes(got)
        out[len(got) :] = b"\x00" * (need - len(got))
        active = bool(self.queue)
        if active != self.playing:
            if not active:
                self.last_play_end = time.monotonic()
            self.playing = active

    def _finish(self, rid):
        self.remaining.pop(rid, None)
        self.done_marks.discard(rid)
        self.started_sent.discard(rid)
        self.loop.call_soon_threadsafe(
            self.send_json, {"type": "playback.ended", "responseId": rid}
        )

    # ---- 主循环侧 ----
    def feed(self, rid, pcm):
        self.queue.append((rid, pcm))
        if rid:
            self.remaining[rid] = self.remaining.get(rid, 0) + len(pcm)

    def mark_done(self, rid):
        if not rid:
            return
        if self.remaining.get(rid, 0) <= 0:
            self.started_sent.discard(rid)
            self.remaining.pop(rid, None)
            self.send_json({"type": "playback.ended", "responseId": rid})
        else:
            self.done_marks.add(rid)

    def clear(self, reason=""):
        pending = {rid for rid, _ in self.queue if rid}
        self.queue.clear()
        for rid in pending:
            self.remaining.pop(rid, None)
            self.done_marks.discard(rid)
            self.started_sent.discard(rid)
            event = {"type": "playback.cancelled", "responseId": rid}
            if reason:
                event["reason"] = reason
            self.send_json(event)

    def start(self):
        self.instream.start()
        self.outstream.start()

    def stop(self):
        for s in (self.instream, self.outstream):
            try:
                s.stop()
                s.close()
            except Exception:
                pass


def ensure_swift_helper():
    """定位(必要时用 swiftc 编译)0.5.0 的 CoreAudio 助手,返回二进制路径。

    查找顺序:QWEN_AUDIO_AGENT_TUI_AEC_BINARY 环境变量 → 用户缓存目录
    (源码 tui/native/macos-voice-io.swift,过期自动重编)→ 脚本同目录。
    """
    env = os.environ.get("QWEN_AUDIO_AGENT_TUI_AEC_BINARY")
    if env and os.access(env, os.X_OK):
        return pathlib.Path(env)
    here = pathlib.Path(__file__).resolve().parent  # tui/fullscreen/
    repo = here.parent.parent
    cache_root = pathlib.Path.home() / "Library/Caches/qwaudio"
    candidates = [
        (cache_root / "tui/macos-voice-io", repo / "tui/native/macos-voice-io.swift"),
        (here / "macos-voice-io", here / "macos-voice-io.swift"),
    ]
    for binary, source in candidates:
        has_bin = binary.exists() and os.access(binary, os.X_OK)
        has_src = source.exists()
        if has_bin and (not has_src or binary.stat().st_mtime >= source.stat().st_mtime):
            return binary
        if has_src and sys.platform == "darwin" and shutil.which("swiftc"):
            binary.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["swiftc", str(source), "-O", "-framework", "AudioToolbox", "-o", str(binary)],
                check=True,
                capture_output=True,
            )
            return binary
    raise FileNotFoundError("未找到 CoreAudio 助手 macos-voice-io")


class SwiftAecEngine:
    """macOS 系统级回声消除引擎:对接 0.5.0 的 CoreAudio 助手,外放全双工。

    协议(JSON 行):
      发 {"type":"play","audio":b64,"sampleRate":24000} / {"type":"clear"}
        {"type":"capture","enabled":bool} / {"type":"close"}
      收 {"type":"ready"} / {"type":"audio","audio":b64}(16k 已消回声)
        {"type":"error","message":...}
    clear 与音频同一条 stdin 流,顺序处理——必然清掉之前发出的全部音频,
    打断没有带外信号与管道积压的竞态。播放回执用时间轴模型排程。
    """

    def __init__(self, loop, send_json, on_level, binary_path):
        self.loop = loop
        self.send_json = send_json
        self.on_level = on_level
        self.proc = subprocess.Popen(
            [str(binary_path), "--capture-rate", str(IN_RATE), "--playback-rate", str(OUT_RATE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.cursor = time.monotonic()  # 播放时间轴(秒)
        self.ends = {}  # rid -> 预计结束时刻
        self.timers = []  # call_later 句柄
        self.started_sent = set()
        self.cancelled = set()  # 被打断的响应:后续块直接丢弃
        self.muted = False
        self._stop = False
        self._ready = asyncio.Event()
        self._error = None
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    @classmethod
    async def create(cls, loop, send_json, on_level, binary_path):
        self = cls(loop, send_json, on_level, binary_path)
        try:
            await asyncio.wait_for(self._ready.wait(), 15)
        except TimeoutError as error:
            self.stop()
            raise RuntimeError(self._error or "CoreAudio 助手启动超时") from error
        self._send({"type": "capture", "enabled": True})
        return self

    @property
    def playing(self):
        return time.monotonic() < self.cursor

    def _send(self, obj):
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            return True
        except (BrokenPipeError, ValueError, OSError):
            return False

    def _read_loop(self):
        for line in self.proc.stdout:
            if self._stop:
                break
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type")
            if kind == "ready":
                self.loop.call_soon_threadsafe(self._ready.set)
            elif kind == "audio" and ev.get("audio"):
                pcm = base64.b64decode(ev["audio"])
                rms = pcm_rms(pcm)
                self.loop.call_soon_threadsafe(self.on_level, min(1.0, rms / 12000), self.playing)
                if not self.muted:
                    self.loop.call_soon_threadsafe(
                        self.send_json, {"type": "audio.append", "audio": ev["audio"]}
                    )
            elif kind == "error":
                self._error = ev.get("message")

    def feed(self, rid, pcm):
        if rid in self.cancelled:
            return  # 该响应已被打断,余下的块全部丢弃
        if not self._send(
            {"type": "play", "sampleRate": OUT_RATE, "audio": base64.b64encode(pcm).decode()}
        ):
            return
        now = time.monotonic()
        start = max(now, self.cursor)
        self.cursor = start + len(pcm) / (OUT_RATE * 2)
        if rid:
            self.ends[rid] = self.cursor
            if rid not in self.started_sent:
                self.started_sent.add(rid)
                self.timers.append(
                    self.loop.call_later(
                        max(0, start - now),
                        lambda r=rid: self.send_json({"type": "playback.started", "responseId": r}),
                    )
                )

    def mark_done(self, rid):
        if not rid:
            return
        end = self.ends.get(rid, time.monotonic())
        self.timers.append(
            self.loop.call_later(
                max(0, end - time.monotonic()) + 0.05, lambda r=rid: self._finish(r)
            )
        )

    def _finish(self, rid):
        self.ends.pop(rid, None)
        self.started_sent.discard(rid)
        self.send_json({"type": "playback.ended", "responseId": rid})

    def clear(self, reason=""):
        # 带内 clear 顺序处理,必然清掉之前的全部音频;晚到块由 cancelled 拦截
        self.cancelled.update(rid for rid in self.ends)
        self._send({"type": "clear"})
        for h in self.timers:
            h.cancel()
        self.timers.clear()
        for rid in list(self.ends):
            event = {"type": "playback.cancelled", "responseId": rid}
            if reason:
                event["reason"] = reason
            self.send_json(event)
        self.ends.clear()
        self.started_sent.clear()
        self.cursor = time.monotonic()

    def start(self):
        pass  # 进程已在跑

    def stop(self):
        self._stop = True
        self._send({"type": "close"})
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def build_app(args):
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Input, RichLog, Static

    STATUS_ZH = {
        "running": "进行中",
        "queued": "排队",
        "completed": "完成",
        "failed": "失败",
        "cancelled": "已取消",
    }

    class QwenAudioAgentApp(App):
        TITLE = "qwen-audio-agent"
        CSS = """
        Screen { layout: vertical; background: $surface; }
        #statebar {
            height: 3; padding: 0 2; content-align: left middle;
            background: $panel; border-bottom: hkey $primary 30%;
        }
        #main { height: 1fr; }
        #conversation { width: 2fr; height: 1fr; }
        #chat {
            width: 1fr; height: 1fr; border: round $primary 40%;
            border-title-align: left;
            padding: 0 1; background: $surface;
        }
        #live {
            display: none; height: auto; max-height: 4; margin: 0 1;
            padding: 0 1; color: $text;
            border-left: thick $primary 60%;
        }
        #tasks {
            width: 1fr; min-width: 34; border: round $secondary 40%;
            border-title-align: left; padding: 0 1; color: $text-muted;
        }
        #inputbox { dock: bottom; border: tall $primary 50%; margin: 0 1 1 1; }
        """
        BINDINGS: ClassVar[list[Binding]] = [
            Binding("ctrl+x", "interrupt", "打断"),
            Binding("ctrl+t", "toggle_mute", "静音"),
            Binding("ctrl+q", "quit", "退出"),
        ]

        def __init__(self):
            super().__init__()
            self.ws = None
            self.http = None
            self.engine = None
            self._user_streams = {}
            self._assistant_streams = {}
            self.mic_level = 0.0
            self.playing = False
            self.state_text = "connecting"
            self.quick_hint = ""
            self._send_tasks: set[asyncio.Task] = set()

        # ---------- 布局 ----------
        def compose(self) -> ComposeResult:
            yield Static(id="statebar")
            with Horizontal(id="main"):
                with Vertical(id="conversation"):
                    chat = RichLog(id="chat", wrap=True, markup=True, auto_scroll=True)
                    chat.border_title = "对话"
                    yield chat
                    yield Static(id="live")
                tasks = Static("空闲", id="tasks")
                tasks.border_title = "任务"
                yield tasks
            yield Input(placeholder="打字发送 · 空回车打断 · /tasks /cancel", id="inputbox")
            yield Footer()

        # ---------- 状态栏 ----------
        def render_statebar(self):
            playing = self.engine.playing if self.engine else False
            if self.state_text == "connecting":
                dot, label, color = "◌", "连接中", "grey62"
            elif playing:
                interrupt_hint = (
                    "开口即可打断" if getattr(self, "engine_kind", "") == "vpio" else "回车打断"
                )
                dot, label, color = "●", f"回答中({interrupt_hint})", "dodger_blue1"
            else:
                dot, label, color = "●", "在听,请讲", "green3"
            bar_n = int(self.mic_level * 24)
            vu = "▁" * 0 + "█" * bar_n + "░" * (24 - bar_n)
            kind = getattr(self, "engine_kind", "none")
            if kind == "vpio":
                mode = "全双工·系统回声消除"
            else:
                mode = (
                    "耳机全双工"
                    if args.headphones
                    else ("外放·能量突破(实验)" if args.barge_in else "外放半双工")
                )
            dbg = ""
            if self.engine and getattr(self.engine, "dbg", {}).get("gated"):
                d = self.engine.dbg
                dbg = f"  [插话检测 音量{d['rms']}/线{d['thr']} 连击{d['run']}]"
            text = Text.assemble(
                (f" {dot} ", color),
                (f"{label}", f"bold {color}"),
                ("   🎤 ", "grey62"),
                (vu, "green3" if not playing else "grey42"),
                (f"   {mode}", "grey54"),
                (f"   会话 {args.session}", "grey42"),
                (f"   {self.quick_hint}", "yellow3"),
                (dbg, "orange3"),
            )
            self.query_one("#statebar", Static).update(text)

        # ---------- 生命周期 ----------
        async def on_mount(self):
            self.set_interval(0.1, self.render_statebar)
            self.set_interval(2.0, self.refresh_tasks)
            self.run_worker(self.connect(), exclusive=False)
            self.query_one("#inputbox", Input).focus()

        async def connect(self):
            chat = self.query_one("#chat", RichLog)
            self.http = aiohttp.ClientSession()
            base = args.url.rstrip("/")
            try:
                health = await (await self.http.get(f"{base}/api/health")).json()
                if not health.get("ok"):
                    chat.write(f"[red]服务未就绪: {health}[/red]")
                    return
                self.ws = await self.http.ws_connect(
                    f"{base}/api/realtime?sessionId={args.session}", heartbeat=20
                )
            except Exception as e:
                chat.write(f"[red]连接失败: {e}[/red]")
                return

            loop = asyncio.get_event_loop()

            def send_json(obj):
                if self.ws and not self.ws.closed:
                    task = asyncio.create_task(self.ws.send_str(json.dumps(obj)))
                    self._send_tasks.add(task)
                    task.add_done_callback(self._send_tasks.discard)

            self._send_json = send_json
            self.engine_kind = "none"
            try:
                binary = ensure_swift_helper()
                self.engine = await SwiftAecEngine.create(
                    loop, send_json, on_level=self._on_level, binary_path=binary
                )
                self.engine_kind = "vpio"
                chat.write("[grey62]· 语音就绪(系统回声消除,全双工:随时开口即可打断)[/grey62]")
            except Exception as e:
                try:
                    self.engine = VoiceEngine(
                        loop,
                        send_json,
                        on_level=self._on_level,
                        on_barge=self._on_barge,
                        mic_gate=not args.headphones,
                        barge_in=args.barge_in,
                    )
                    self.engine.start()
                    self.engine_kind = "portaudio"
                    chat.write(
                        f"[grey62]· 语音就绪(半双工;{e};"
                        f"macOS 装 Xcode 命令行工具可升级全双工)[/grey62]"
                    )
                except Exception as e2:
                    chat.write(f"[yellow]音频不可用({e2}),仅文字模式[/yellow]")

            send_json(
                {
                    "type": "connect",
                    "voiceEnabled": True,
                    "inputEnabled": not bool(self.engine and self.engine.muted),
                    "outputEnabled": True,
                    "clientType": "cli",
                    "clientLabel": "CLI",
                    "takeover": args.takeover,
                    "timeZone": "Asia/Shanghai",
                    "locale": "zh-CN",
                }
            )
            self.state_text = "ready"
            await self.reader()

        def _on_level(self, level, playing):
            self.mic_level = level

        def _on_barge(self):
            # 能量突破:用户在播放期大声说话 → 立即打断并临时放行麦克风
            if not (self.engine and self.engine.playing):
                return
            self.engine.clear("user_interruption")
            self.engine.last_play_end = 0.0  # 立刻解除门控尾窗
            self._send_json({"type": "interrupt"})
            self.query_one("#chat", RichLog).write("[grey62]· (检测到插话,已打断)[/grey62]")

        # ---------- 下行 ----------
        async def reader(self):
            chat = self.query_one("#chat", RichLog)
            async for msg in self.ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                try:
                    m = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                mt = m.get("type")
                if mt == "audio.delta" and self.engine:
                    self.engine.feed(
                        str(m.get("responseId") or ""), base64.b64decode(m.get("audio") or "")
                    )
                elif mt == "audio.done" and self.engine:
                    self.engine.mark_done(str(m.get("responseId") or ""))
                elif mt == "playback.clear" and self.engine:
                    self.engine.clear(str(m.get("reason") or ""))
                elif mt == "voice.deactivated":
                    if self.engine:
                        self.engine.muted = True
                        self.engine.clear()
                    self._clear_stream()
                    chat.write("[yellow]· 语音已切换到另一窗口[/yellow]")
                elif mt == "voice.ownership" and m.get("state") == "busy":
                    if self.engine:
                        self.engine.muted = True
                        self.engine.clear()
                    holder = frontend_label(m.get("holder"))
                    chat.write(
                        f"[yellow]· 语音正由{holder}使用;"
                        "如需接管,请运行 qwenaudio --takeover[/yellow]"
                    )
                elif mt == "transcript.delta" and m.get("role") == "user":
                    turn_id = str(m.get("turnId") or "")
                    incoming = m.get("content") or ""
                    text = (
                        incoming
                        if m.get("replace") is True
                        else self._user_streams.get(turn_id, "") + incoming
                    )
                    self._user_streams[turn_id] = text
                    self._show_stream("你", text, "bold cyan")
                elif mt == "transcript.final" and m.get("role") == "user":
                    turn_id = str(m.get("turnId") or "")
                    text = complete_transcript(
                        self._user_streams.pop(turn_id, ""), m.get("content")
                    )
                    self._clear_stream()
                    chat.write(Text.assemble(("你", "bold cyan"), (f"  {text}", "")))
                elif mt == "transcript.discard" and m.get("role") == "user":
                    self._user_streams.pop(str(m.get("turnId") or ""), None)
                    self._clear_stream()
                elif mt == "transcript.delta" and m.get("role") == "assistant":
                    rid = str(m.get("responseId") or "")
                    text = self._assistant_streams.get(rid, "")
                    text += m.get("content") or ""
                    self._assistant_streams[rid] = text
                    self._show_stream("qwen-audio", text, "bold")
                elif mt == "transcript.final" and m.get("role") == "assistant":
                    rid = str(m.get("responseId") or "")
                    text = complete_transcript(
                        self._assistant_streams.pop(rid, ""), m.get("content")
                    )
                    self._clear_stream()
                    if text:
                        chat.write(Text.assemble(("qwen-audio", "bold"), (f"  {text}", "")))
                elif mt == "error":
                    self._clear_stream()
                    chat.write(f"[red]错误: {m.get('message')}[/red]")
            chat.write("[red]连接已断开(Ctrl+Q 退出后重连)[/red]")
            self.state_text = "connecting"

        def _show_stream(self, role, content, style):
            live = self.query_one("#live", Static)
            live.display = True
            live.update(Text.assemble((role, style), (f"  {content}", "")))

        def _clear_stream(self):
            live = self.query_one("#live", Static)
            live.update("")
            live.display = False

        # ---------- 任务面板 ----------
        async def refresh_tasks(self):
            if not self.http:
                return
            base = args.url.rstrip("/")
            try:
                data = await (
                    await self.http.get(f"{base}/api/tasks?sessionId={args.session}")
                ).json()
            except Exception:
                return
            tasks = data.get("tasks") or []
            widget = self.query_one("#tasks", Static)
            if not tasks:
                widget.update("[grey42]空闲[/grey42]")
                return
            lines = []
            for t in tasks[:8]:
                st = t.get("status", "")
                color = {
                    "running": "yellow3",
                    "completed": "green3",
                    "failed": "red",
                    "pending": "grey62",
                }.get(st, "grey62")
                secs = max(0, round((t.get("elapsedMs") or 0) / 1000))
                obj = (t.get("objective") or "")[:34]
                lines.append(f"[{color}]● {STATUS_ZH.get(st, st)}[/{color}] {secs}s\n  {obj}")
            widget.update("\n".join(lines))

        # ---------- 输入与按键 ----------
        async def on_input_submitted(self, event):
            text = event.value.strip()
            self.query_one("#inputbox", Input).value = ""
            chat = self.query_one("#chat", RichLog)
            if not text:
                self.action_interrupt()
                return
            if text.startswith("/"):
                await self.command(text, chat)
                return
            self._send_json({"type": "text.message", "text": text})

        async def command(self, line, chat):
            parts = line.split()
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
            base = args.url.rstrip("/")
            if cmd == "/quit":
                await self.action_quit()
            elif cmd == "/tasks":
                await self.refresh_tasks()
                chat.write("[grey62]· 任务面板已刷新(右侧)[/grey62]")
            elif cmd == "/cancel":
                data = await (
                    await self.http.get(f"{base}/api/tasks?sessionId={args.session}")
                ).json()
                tasks = data.get("tasks") or []
                target = (
                    next((t for t in tasks if t.get("id") == arg), None)
                    if arg
                    else next((t for t in tasks if t.get("status") in ("running", "queued")), None)
                )
                if not target:
                    chat.write("[red]没有可取消的任务[/red]")
                    return
                await self.http.delete(f"{base}/api/tasks/{target['id']}")
                chat.write(f"[grey62]· 已取消 {target['id']}[/grey62]")
            else:
                chat.write(f"[red]未知命令 {cmd}[/red]")

        def action_interrupt(self):
            if self.engine:
                self.engine.clear("user_interruption")
            self._send_json({"type": "interrupt"})

        def action_toggle_mute(self):
            if not self.engine:
                return
            self.engine.muted = not self.engine.muted
            self._send_json(
                {
                    "type": "input.mute" if self.engine.muted else "input.unmute",
                    **({} if self.engine.muted else {"takeover": False}),
                }
            )
            self.query_one("#chat", RichLog).write(
                "[grey62]· 麦克风已静音[/grey62]"
                if self.engine.muted
                else "[grey62]· 麦克风已恢复[/grey62]"
            )

        async def on_unmount(self):
            if self.engine:
                self.engine.stop()
            for task in self._send_tasks:
                task.cancel()
            await asyncio.gather(*self._send_tasks, return_exceptions=True)
            if self.ws:
                await self.ws.close()
            if self.http:
                await self.http.close()

    return QwenAudioAgentApp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:3101")
    ap.add_argument("--session", default="voice-main")
    ap.add_argument("--takeover", action="store_true", help="明确接管其他前端的语音控制权")
    ap.add_argument(
        "--headphones", action="store_true", help="耳机模式:关闭防回声门控,全双工语音打断"
    )
    ap.add_argument(
        "--barge-in",
        dest="barge_in",
        action="store_true",
        help="实验:外放下能量突破语音打断(含诊断数字)",
    )
    args = ap.parse_args()
    build_app(args).run()


if __name__ == "__main__":
    main()
