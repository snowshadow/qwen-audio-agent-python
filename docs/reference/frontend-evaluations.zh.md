# 前台 Runtime 评测

原先针对已下线 JavaScript Gateway 的 `eval:frontend` 套件已移除。

当前 Python 测试（`tests/unit`、`tests/integration`）覆盖活 Gateway 在不依赖
Realtime 模型时也能保证的边界：

| 维度 | 对应测试 |
| --- | --- |
| 前台工具路由 | `tests/unit/test_frontend_graph.py`、`tests/unit/test_work_graph.py` |
| 打断与回合隔离 | `tests/unit/test_turns.py`、`tests/integration/test_realtime_application.py` |
| 结果交付与重复呈现 | `tests/unit/test_result_delivery.py`、`tests/unit/test_announcement_window.py` |
| Work 生命周期 | `tests/integration/test_work_runtime.py`、`tests/integration/test_work_recovery.py` |

Search、Knowledge 与 Citation 身份评测属于尚未接入 Python Gateway 的扩展点。
见[架构文档](../architecture.zh.md#当前-v2-实现)。

运行：

```bash
npm run test:python
```
