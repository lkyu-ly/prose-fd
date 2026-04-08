from __future__ import annotations

import py_compile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PADDLE_ROOT = REPO_ROOT / "prose_fd_paddle"
FAILED_MARKER = ">" * 7
FORBIDDEN_DADAPT_IMPORT = "from " + "dada" + "ptation" + " import"
FORBIDDEN_TORCHDATA_IMPORT = "import " + "torch" + "data.data" + "pipes"
FORBIDDEN_ROTARY_IMPORT = "from " + "rotary_" + "embedding_torch" + " import"
FORBIDDEN_TOKENS = (
    FAILED_MARKER,
    FORBIDDEN_DADAPT_IMPORT,
    FORBIDDEN_TORCHDATA_IMPORT,
    FORBIDDEN_ROTARY_IMPORT,
)
FORBIDDEN_LINE_PREFIXES = (
    "import torch",
    "from torch ",
    "from torch.",
)


def iter_runtime_python_files():
    for path in sorted(PADDLE_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        yield path


def test_runtime_tree_has_no_forbidden_torch_residuals():
    offenders: list[str] = []
    for path in iter_runtime_python_files():
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in FORBIDDEN_TOKENS):
            offenders.append(str(path.relative_to(REPO_ROOT)))
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(FORBIDDEN_LINE_PREFIXES):
                offenders.append(str(path.relative_to(REPO_ROOT)))
                break
    assert not offenders, f"runtime files still contain forbidden residuals: {offenders}"


def test_runtime_tree_is_python_syntax_clean():
    failures: list[str] = []
    for path in iter_runtime_python_files():
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(REPO_ROOT)}: {exc.msg}")
    assert not failures, "python syntax errors remain:\n" + "\n".join(failures)
