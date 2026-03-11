#!/usr/bin/env python3
"""
sync_deps.py — Scans Python source files and notebooks for third-party imports,
resolves canonical PyPI names, and outputs a sorted dependency list to stdout.

Usage:
    python sync_deps.py [root_dir]

Output (one line per third-party package, sorted):
    package-name>=X.Y

Design invariants:
- IMPORT_TO_PACKAGE is the ONLY place to fix import-name → PyPI-name mismatches.
- DEV_ONLY_PACKAGES lists packages excluded from requirements.txt (dev-only).
- EXCLUDE_DIRS lists directory names never scanned.
- No side effects: stdout is the only output; nothing is written to disk.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these constants; do not hardcode mappings elsewhere.
# ---------------------------------------------------------------------------

# Maps Python import root name → canonical PyPI package name.
# Only entries that differ between the import name and the PyPI name are listed.
IMPORT_TO_PACKAGE: dict[str, str] = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "faiss": "faiss-cpu",
    "rank_bm25": "rank-bm25",
    "langchain_classic": "langchain-classic",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_openai": "langchain-openai",
    "langchain": "langchain",   # venv-specific: Anthropic Agent SDK
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "google.cloud": "google-cloud",
    "google.auth": "google-auth",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "serial": "pyserial",
    "usb": "pyusb",
    "gi": "PyGObject",
    "wx": "wxPython",
    "gtk": "PyGTK",
}

# Packages excluded from requirements.txt (dev / build tools only).
DEV_ONLY_PACKAGES: frozenset[str] = frozenset(
    {
        "pytest", "pytest_cov", "pytest-cov",
        "mypy", "ruff", "black", "isort",
        "pylint", "flake8", "bandit",
        "build", "twine", "setuptools", "wheel", "pip",
        "tox", "nox", "pre-commit",
        "ipykernel", "ipython", "jupyter", "notebook", "jupyterlab",
    }
)

# Directory names never scanned for imports.
EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "venv", ".venv", "env", "ENV",
        "build", "dist", ".eggs",
        "__pycache__", ".git", ".tox",
        ".mypy_cache", ".ruff_cache", ".pytest_cache",
        "node_modules",
    }
)


# ---------------------------------------------------------------------------
# Stdlib detection (Python 3.10+ has sys.stdlib_module_names; fallback below)
# ---------------------------------------------------------------------------

def _stdlib_modules() -> frozenset[str]:
    if hasattr(sys, "stdlib_module_names"):
        return frozenset(sys.stdlib_module_names)
    # Minimal static fallback for older Python versions.
    return frozenset(
        {
            "abc", "ast", "asyncio", "builtins", "collections", "contextlib",
            "copy", "csv", "dataclasses", "datetime", "enum", "functools",
            "glob", "hashlib", "html", "http", "importlib", "inspect",
            "io", "itertools", "json", "logging", "math", "multiprocessing",
            "operator", "os", "pathlib", "pickle", "platform", "pprint",
            "queue", "random", "re", "shutil", "signal", "socket", "sqlite3",
            "ssl", "stat", "string", "struct", "subprocess", "sys",
            "tempfile", "textwrap", "threading", "time", "timeit", "traceback",
            "typing", "unittest", "urllib", "uuid", "warnings", "weakref",
            "xml", "xmlrpc", "zipfile", "zlib",
            # common sub-packages
            "collections.abc", "os.path", "urllib.parse", "urllib.request",
        }
    )


STDLIB: frozenset[str] = _stdlib_modules()


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIRS for part in path.parts)


def collect_local_modules(root: Path) -> frozenset[str]:
    """
    Returns names of importable modules defined locally in the project root.
    These are excluded from the dep list because they are not third-party packages.
    Covers:
    - Top-level .py files   (e.g. utils.py  → 'utils')
    - Top-level packages    (e.g. kitai/    → 'kitai')
    """
    local: set[str] = set()
    for child in root.iterdir():
        if child.is_file() and child.suffix == ".py" and not _is_excluded(child):
            local.add(child.stem)
        elif child.is_dir() and (child / "__init__.py").exists() and not _is_excluded(child):
            local.add(child.name)
    return frozenset(local)


def collect_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if not _is_excluded(p)]


def collect_notebooks(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.ipynb") if not _is_excluded(p)]


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def _extract_from_ast(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:   # absolute import only
                names.add(node.module.split(".")[0])
    return names


def extract_imports_from_py(path: Path) -> set[str]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        return _extract_from_ast(tree)
    except (SyntaxError, ValueError):
        return set()


def extract_imports_from_notebook(path: Path) -> set[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        names: set[str] = set()
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source)
                names |= _extract_from_ast(tree)
            except (SyntaxError, ValueError):
                continue
        return names
    except (json.JSONDecodeError, KeyError):
        return set()


# ---------------------------------------------------------------------------
# Package resolution
# ---------------------------------------------------------------------------

def get_installed_versions() -> dict[str, str]:
    """Returns {normalised_name: version} for all packages in the current venv."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    packages: list[dict] = json.loads(result.stdout)
    # normalise: lowercase, replace hyphens/dots with underscores
    return {
        pkg["name"].lower().replace("-", "_").replace(".", "_"): pkg["version"]
        for pkg in packages
    }


def _normalise(name: str) -> str:
    """Normalise a package name for version lookup."""
    return name.lower().replace("-", "_").replace(".", "_")


def resolve_packages(
    raw_imports: set[str],
    own_package: str,
    installed: dict[str, str],
    local_modules: frozenset[str],
) -> dict[str, str]:
    """
    Filter raw imports to third-party only, map to PyPI names, attach versions.

    Returns:
        dict mapping PyPI package name → version string (empty if not installed).
    """
    result: dict[str, str] = {}
    own_norm = _normalise(own_package)

    for imp in sorted(raw_imports):
        if (
            imp in STDLIB
            or imp.startswith("_")
            or _normalise(imp) == own_norm
            or _normalise(imp) in DEV_ONLY_PACKAGES
            or imp in local_modules           # skip local .py files / packages
        ):
            continue

        pypi = IMPORT_TO_PACKAGE.get(imp, imp.replace("_", "-"))

        if _normalise(pypi) in DEV_ONLY_PACKAGES:
            continue

        # Try several normalisation forms to find the version.
        version = (
            installed.get(_normalise(pypi))
            or installed.get(_normalise(imp))
            or ""
        )
        result[pypi] = version

    return result


# ---------------------------------------------------------------------------
# pyproject.toml: detect own package name
# ---------------------------------------------------------------------------

def _own_package_name(root: Path) -> str:
    toml_path = root / "pyproject.toml"
    if toml_path.exists():
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            name: str = data.get("project", {}).get("name", "")
            return name.replace("-", "_")
        except Exception:
            pass
    return root.name.replace("-", "_")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()

    if not root.is_dir():
        print(f"error: '{root}' is not a directory", file=sys.stderr)
        sys.exit(1)

    own_package = _own_package_name(root)
    local_modules = collect_local_modules(root)

    # 1. Collect files.
    py_files = collect_py_files(root)
    nb_files = collect_notebooks(root)

    print(
        f"# Scanned {len(py_files)} .py files and {len(nb_files)} notebooks "
        f"(root: {root})",
        file=sys.stderr,
    )

    # 2. Extract all imports.
    raw: set[str] = set()
    for f in py_files:
        raw |= extract_imports_from_py(f)
    for f in nb_files:
        raw |= extract_imports_from_notebook(f)

    # 3. Resolve.
    installed = get_installed_versions()
    packages = resolve_packages(raw, own_package, installed, local_modules)

    print(f"# Found {len(packages)} third-party packages", file=sys.stderr)

    # 4. Output: one line per package, sorted case-insensitively.
    for name, version in sorted(packages.items(), key=lambda kv: kv[0].lower()):
        if version:
            # Pin to major.minor for flexibility.
            major_minor = ".".join(version.split(".")[:2])
            print(f"{name}>={major_minor}")
        else:
            # Package not installed locally — output name only as a reminder.
            print(f"{name}  # version unknown — install first")


if __name__ == "__main__":
    main()
