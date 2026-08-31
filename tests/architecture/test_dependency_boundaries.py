from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "qwen_audio_agent"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def assert_no_import_prefix(path: Path, forbidden: tuple[str, ...]) -> None:
    imports = imported_modules(path)
    violations = sorted(
        module for module in imports if any(module.startswith(prefix) for prefix in forbidden)
    )
    message = f"{path.relative_to(PACKAGE_ROOT)} imports forbidden modules: {violations}"
    assert not violations, message


def test_protocol_is_dependency_free_from_runtime_layers() -> None:
    forbidden = (
        "qwen_audio_agent.backend",
        "qwen_audio_agent.domain",
        "qwen_audio_agent.graphs",
        "qwen_audio_agent.runtime",
        "qwen_audio_agent.transport",
    )
    for path in (PACKAGE_ROOT / "protocol").rglob("*.py"):
        assert_no_import_prefix(path, forbidden)


def test_domain_does_not_depend_on_transport_or_adapters() -> None:
    forbidden = (
        "qwen_audio_agent.transport",
        "qwen_audio_agent.protocol",
        "qwen_audio_agent.graphs",
        "qwen_audio_agent.runtime",
    )
    for path in (PACKAGE_ROOT / "domain").rglob("*.py"):
        assert_no_import_prefix(path, forbidden)


def test_frontend_graph_does_not_depend_on_backend_runtime() -> None:
    path = PACKAGE_ROOT / "graphs" / "frontend.py"
    if path.exists():
        assert_no_import_prefix(path, ("qwen_audio_agent.backend",))


def test_work_graph_does_not_depend_on_transport() -> None:
    forbidden = ("qwen_audio_agent.transport",)
    for name in ("work.py", "work_graph.py"):
        path = PACKAGE_ROOT / "graphs" / name
        if path.exists():
            assert_no_import_prefix(path, forbidden)
