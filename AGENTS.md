# AGENTS.md

## Project context

This repository uses `reports/context/experiment_log.md` as the persistent source of truth for daily reports, weekly progress PPTs, and monthly plans.

The `experiment_log.md` file is not a polished report. It is a factual project memory.

Formal report formatting, such as daily report format, weekly PPT structure, and monthly plan tables, should be handled later by the `$report-assistant` skill.

## Automatic experiment logging

After every meaningful Codex task, append a clear and accurate summary to:

`reports/context/experiment_log.md`

Create the directory and file if they do not exist.

A meaningful task includes:

* answering a project-related technical question
* modifying code
* debugging
* running commands
* changing configs
* running experiments or evaluations
* analyzing logs, results, errors, metrics, or outputs
* preparing research content, reports, plans, or PPT material

## What to write

Write a concise but complete factual summary of what happened in the current task.

The summary should include relevant details such as:

* what problem or task was addressed
* what files, scripts, configs, or commands were involved
* what changes were made
* what experiments or checks were run
* what results, errors, metrics, or observations were obtained
* what conclusions are supported by the evidence
* what remains unresolved
* what should be done next
  

Do not force the summary into a daily-report template.

Do not use fixed sections like “今日完成 / 卡点 / 明日计划” unless the user explicitly asks for a daily report.

## Logging rules

* Only record facts from the current task.
* Do not invent results, success rates, speedups, or conclusions.
* If a result is not available, say it is not available.
* Preserve important paths, config names, command names, checkpoint paths, metric names, and metric values.
* Keep the writing clear and easy to reuse later.
* Append only. Never overwrite existing content.
* If the task is trivial and adds no useful project information, logging can be skipped.

## Report generation

When the user asks for a daily report, weekly report PPT, or monthly plan, use the `$report-assistant` skill and read:

`reports/context/experiment_log.md`

The skill is responsible for transforming the factual log into the required final format.
