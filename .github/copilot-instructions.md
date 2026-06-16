# Busch_2024 Workspace Instructions

Use this workspace instruction file for work in the Busch_2024 repository.

## Default context sequence

Before doing substantial work, use the knowledge base at `C:\Users\johnr\Documents\JFR_knowledge_base` as the first context layer.

Read in this order when relevant:

1. `C:\Users\johnr\Documents\JFR_knowledge_base\03_indexes\Start Here.md`
2. `C:\Users\johnr\Documents\JFR_knowledge_base\03_indexes\index.md`
3. The relevant domain index
4. The relevant glossary, lineage, workflow, or source-summary notes

Then verify any important code or data claim against the live source files in this repository before editing.

## Working preferences

- Prefer authoritative source tracing over inference from downstream outputs or derived databases.
- For Busch logic questions, treat the Stata files in `DO code/` as the methodological source of truth unless the task explicitly concerns a Python-only layer.
- In this workspace, the Stata `.do` files are typically used as reference for methodology and calculation logic, not as scripts to run or debug, unless the task explicitly says otherwise.
- File durable findings back into the knowledge base when the result is likely to matter in later sessions.

## Tool use on Windows

- This repository is often worked on from Windows and PowerShell. Do not assume Unix command-line tools are installed.
- Prefer built-in editor and agent tools for search, file reads, diagnostics, and edits when available.
- If terminal search is needed and `rg` is installed, prefer it. If `rg` is not installed, use PowerShell-native alternatives such as `Get-ChildItem`, `Select-String`, and `Test-Path`.
- Prefer Windows-appropriate commands and PowerShell syntax when using the terminal in this workspace.
- If a missing tool would materially improve work in this repository, explicitly invite the user to install it rather than silently struggling without it.

## Python style

When editing Python in this repository, follow the style implied by `C:\Users\johnr\Documents\Work documents\3 Programming utilities\Python\Make_it_JFR_style.py` when feasible:

- compact formatting
- restrained blank lines
- dense but readable layout
- preserve correctness and local project conventions first