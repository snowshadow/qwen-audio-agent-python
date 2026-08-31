# Frontend Runtime Evaluations

The previous JavaScript `eval:frontend` suite targeted the retired JS Gateway
runtime and has been removed.

Live Python tests under `tests/unit` and `tests/integration` cover the
invariants the current Gateway can guarantee without a Realtime model:

| Dimension | Where it is checked |
| --- | --- |
| Frontend tool routing | `tests/unit/test_frontend_graph.py`, `tests/unit/test_work_graph.py` |
| Interruption and turn fencing | `tests/unit/test_turns.py`, `tests/integration/test_realtime_application.py` |
| Result delivery and duplicate presentation | `tests/unit/test_result_delivery.py`, `tests/unit/test_announcement_window.py` |
| Work lifecycle | `tests/integration/test_work_runtime.py`, `tests/integration/test_work_recovery.py` |

Search, Knowledge, and citation-identity evaluations belonged to extension
points that are not connected on the Python Gateway path. See
[architecture](../architecture.md#current-v2-implementation).

Run:

```bash
npm run test:python
```
