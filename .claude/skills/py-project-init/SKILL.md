---
name: py-project-init
description: 'Scaffolds a complete Python library project from scratch. Use when asked to "create a new Python project", "scaffold a Python library", "init a Python project", "set up a new Python package", "create a Python project structure", or "bootstrap a Python project". Asks for the project name, then creates the full directory layout: package folder, scripts/, data/, tests/, .gitignore (venv, .env, AI dev files), requirements.txt, pyproject.toml, README.md, and a virtual environment.'
---

# Python Project Init Skill

Bootstraps a complete, ready-to-code Python library project with a single command.
Asks for the project name, then generates the full directory and file structure in the
current working directory.

## When to Use This Skill

- User says "create a new Python project", "scaffold a Python library", "init a Python project"
- User says "set up a new Python package", "bootstrap a Python project"
- User says "create a Python project structure" or "make a new Python repo"

---

## Step 1 — Ask for the project name

**Always ask before creating anything.** Use `AskUserQuestion` with a single question:

```
Question : "What is the name of the new project?"
Header   : "Project name"
Options  : (none required — user types freely via the Other input)
```

Rules for the name the user provides:
- Convert spaces to hyphens for the **directory name** (`my-project`)
- Convert hyphens/spaces to underscores for the **Python package name** (`my_project`)
- Lowercase both forms

Example: user says `"Kitai RAG"` →
- directory  : `kitai-rag/`
- package    : `kitai_rag/`
- display    : `Kitai RAG`

---

## Step 2 — Confirm the target location

Print the full path of where the project will be created (current working directory + project name) and proceed — do not ask again.

---

## Step 3 — Create the directory tree

Create every folder and file below. Use `python -c "import os; os.makedirs(...)"` for
directories (the environment may not have `mkdir`) and the `Write` tool for all files.

```
{project-name}/
├── {package_name}/
│   └── __init__.py
├── scripts/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── tests/
│   ├── __init__.py
│   └── test_{package_name}.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

The virtual environment (`venv/`) is created in Step 5 — not here.

---

## Step 4 — Write each file

### `{package_name}/__init__.py`

```python
"""
{Display Name} — top-level package.
"""

__version__ = "0.1.0"
```

### `tests/__init__.py`

Empty file (zero bytes). Use `Write` with an empty string.

### `tests/test_{package_name}.py`

```python
"""Basic smoke tests for {display_name}."""

import {package_name}


def test_version():
    assert isinstance({package_name}.__version__, str)
    assert len({package_name}.__version__) > 0
```

### `scripts/.gitkeep` and `data/.gitkeep`

Empty files. Use `Write` with an empty string.

### `.env.example`

```
# Copy this file to .env and fill in real values.
# Never commit .env to version control.

# Example:
# OPENAI_API_KEY=sk-...
# DATABASE_URL=postgresql://user:pass@localhost/dbname
```

### `.gitignore`

```gitignore
# ── Virtual environments ──────────────────────────────────────────────────────
venv/
.venv/
env/
ENV/

# ── Environment / secrets ─────────────────────────────────────────────────────
.env
*.env
.env.local
.env.*.local
secrets.json
credentials.json

# ── AI development tools ──────────────────────────────────────────────────────
# Claude Code
.claude/
# Cursor
.cursor/
.cursorignore
.cursorindexingignore
# GitHub Copilot
.github/copilot-instructions.md
# Windsurf
.windsurf/
# Codeium
.codeium/
# Aider
.aider*
# Continue
.continue/

# ── Python build artefacts ────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg
*.egg-info/
dist/
build/
.eggs/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
*.log

# ── Jupyter ───────────────────────────────────────────────────────────────────
.ipynb_checkpoints/

# ── OS artefacts ──────────────────────────────────────────────────────────────
.DS_Store
Thumbs.db
desktop.ini

# ── Data (keep folder, ignore contents by default) ───────────────────────────
data/*
!data/.gitkeep
```

### `requirements.txt`

```
# Runtime dependencies — pin versions for reproducibility.
# Install with: pip install -r requirements.txt
#
# Example:
# langchain-core>=0.3
# numpy>=1.26
# pandas>=2.2
```

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "{project-name}"
version = "0.1.0"
description = "A short description of {display_name}."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]

dependencies = [
    # Add runtime dependencies here
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov",
    "mypy",
    "ruff",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts   = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
```

### `README.md`

```markdown
# {Display Name}

> Short one-line description.

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```python
import {package_name}

# ...
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project structure

```
{project-name}/
├── {package_name}/     # main package
├── scripts/            # utility scripts
├── data/               # data files (gitignored by default)
├── tests/              # test suite
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

MIT
```
````

---

## Step 5 — Create the virtual environment

Run **inside the new project directory**:

```python
import subprocess, sys, os

project_dir = os.path.join(os.getcwd(), "{project-name}")
subprocess.run(
    [sys.executable, "-m", "venv", os.path.join(project_dir, "venv")],
    check=True,
)
```

Report success with the path to the activate script:
- Windows : `venv\Scripts\activate`
- macOS/Linux : `source venv/bin/activate`

---

## Step 6 — Init git repository

```bash
cd {project-name} && git init && git add . && git commit -m "chore: initial project scaffold"
```

Use `python -c "import subprocess; ..."` if `cd` + `&&` chains are unavailable in the shell.
If the commit fails (no git user configured), skip the commit and tell the user to run it manually.

---

## Step 7 — Print the summary

After all steps succeed, print a clean summary:

```
Project created: /absolute/path/to/{project-name}/

Structure:
  {project-name}/
  ├── {package_name}/    ← Python package
  ├── scripts/           ← utility scripts
  ├── data/              ← data files (gitignored)
  ├── tests/             ← pytest test suite
  ├── venv/              ← virtual environment (gitignored)
  ├── .env.example       ← copy to .env for secrets
  ├── .gitignore
  ├── pyproject.toml
  ├── requirements.txt
  └── README.md

Next steps:
  cd {project-name}
  venv\Scripts\activate        (Windows)
  source venv/bin/activate     (macOS/Linux)
  pip install -r requirements.txt
```

---

## File template reference

### Placeholder substitution table

| Placeholder | Example input `"My RAG Project"` | Result |
|---|---|---|
| `{project-name}` | directory / package name (hyphens) | `my-rag-project` |
| `{package_name}` | Python import name (underscores) | `my_rag_project` |
| `{Display Name}` | human-readable title | `My RAG Project` |

Apply substitutions to every generated file before writing.

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `mkdir` not found | Shell is minimal (no coreutils) | Use `python -c "import os; os.makedirs(...)"` |
| `venv` creation fails | Python not in PATH | Use `sys.executable` to get the current Python path |
| Git commit fails on "no identity" | No git user configured globally | Skip commit; show user `git config --global user.email` instructions |
| Package name collision | Project name matches a stdlib module | Warn the user and suggest a different name |
| `pyproject.toml` build backend error | Old setuptools | Note requires `setuptools>=68`; suggest `pip install -U setuptools` |
