---
inclusion: always
---

# Coding Standards

## Language and Environment
- All code is Python only.
- Always use a virtual environment (`venv`). The project venv is at `venv/` in the workspace root.
- Dependencies are managed via `requirements.txt`.

## Comments
- Comment only when the why is not obvious from the code itself.
- Never comment what the code does — only why it does it if non-obvious.
- No section dividers, no decorative comments, no redundant docstrings that restate the function signature.

## Style
- Never use emojis anywhere in source code, comments, docstrings, log messages, print statements, or CLI output.
- No emoji in any string literal that ends up in code files.
- Follow PEP 8. Use type hints throughout.
- Keep functions small and focused. Prefer explicit over clever.

## Research-Grade Output
- The implementation must produce results suitable for drafting and supporting a research paper.
- Metrics, plots, and CSV outputs must be publication-quality: labeled axes, consistent formatting, reproducible via seed.
- All experiment results must be fully reproducible given the same config and seed.
