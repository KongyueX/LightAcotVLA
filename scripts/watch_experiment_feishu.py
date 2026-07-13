"""Send a signed Feishu notification when a remote experiment finishes."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", "--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--pid", type=int, default=None, help="Experiment PID to monitor.")
    parser.add_argument("--poll_seconds", "--poll-seconds", type=int, default=900)
    parser.add_argument(
        "--progress_seconds",
        "--progress-seconds",
        type=int,
        default=0,
        help="If positive, send a running progress update at this interval.",
    )
    parser.add_argument("--mode", default="full", help="Aggregate mode to summarize from summary.json.")
    parser.add_argument("--label", default="ACoT-VLA experiment")
    parser.add_argument("--test_message", "--test-message", default=None)
    parser.add_argument("--webhook_env", default="FEISHU_WEBHOOK_URL")
    parser.add_argument("--secret_env", default="FEISHU_SIGNING_SECRET")
    args = parser.parse_args()
    if args.pid is None and args.test_message is None:
        parser.error("--pid is required unless --test_message is used.")
    if args.poll_seconds <= 0:
        parser.error("--poll_seconds must be positive.")
    if args.progress_seconds < 0:
        parser.error("--progress_seconds must be non-negative.")
    return args


def _signed_payload(text: str, secret: str) -> dict[str, Any]:
    timestamp = int(time.time())
    string_to_sign = f"{timestamp}\n{secret}"
    signature = base64.b64encode(
        hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    ).decode("utf-8")
    return {
        "timestamp": timestamp,
        "sign": signature,
        "msg_type": "text",
        "content": {"text": text},
    }


def _send_feishu(text: str, *, webhook_url: str, secret: str) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(_signed_payload(text, secret)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Feishu webhook HTTP {exc.code}: {body}") from exc

    code = result.get("code", result.get("StatusCode", 0))
    if code != 0:
        message = result.get("msg", result.get("StatusMessage", "unknown error"))
        raise RuntimeError(f"Feishu webhook rejected the message: code={code}, message={message}")


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _format_percent(value: Any) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number * 100.0:.2f}%"


def _format_number(value: Any, digits: int = 2) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _format_seconds(value: Any) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number / 1000.0:.2f}s"


def _summary_message(output_dir: pathlib.Path, *, label: str, mode: str) -> str:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("base_policy_frozen") is not None:
        metrics = summary.get("last_train_metrics", {})
        return "\n".join(
            [
                f"[ACoT-VLA] {label} completed",
                "kind: execution-horizon SFT",
                f"base policy frozen: {summary.get('base_policy_frozen')}",
                f"records: {summary.get('num_records', 'n/a')}",
                f"steps: {summary.get('train_steps', 'n/a')}",
                f"final loss: {_format_number(metrics.get('train/loss'), 4)}",
                f"elapsed: {_format_number(_finite_number(summary.get('elapsed_seconds')), 1)}s",
                f"params: {summary.get('predictor_params', 'n/a')}",
                f"output: {output_dir}",
            ]
        )
    if "overall" in summary:
        metrics = summary["overall"].get(mode)
        if metrics is None and summary["overall"]:
            mode, metrics = next(iter(summary["overall"].items()))
        metrics = metrics or {}
        return "\n".join(
            [
                f"[ACoT-VLA] {label} completed",
                f"mode: {mode}",
                f"success: {_format_percent(metrics.get('success_rate'))}",
                f"timeout: {_format_percent(metrics.get('timeout_rate'))}",
                f"calls/episode: {_format_number(metrics.get('calls_per_episode'))}",
                f"avg H: {_format_number(metrics.get('avg_h'))}",
                "actual policy/episode: " + _format_seconds(metrics.get("actual_policy_ms_per_episode")),
                "actual wall/episode: " + _format_seconds(metrics.get("actual_wall_ms_per_episode")),
                "predictor/episode: " + _format_seconds(metrics.get("predictor_ms_per_episode")),
                f"H distribution: {json.dumps(metrics.get('h_distribution', {}), sort_keys=True)}",
                f"output: {output_dir}",
            ]
        )
    if "num_records" in summary:
        return "\n".join(
            [
                f"[ACoT-VLA] {label} completed",
                "kind: counterfactual collection",
                f"records: {summary.get('num_records')}",
                f"teacher K: {summary.get('teacher_samples', 'n/a')}",
                f"branch success/H: {summary.get('branch_success_rate_by_h', 'n/a')}",
                f"elapsed: {_format_number(_finite_number(summary.get('elapsed_seconds')), 1)}s",
                f"output: {output_dir}",
            ]
        )
    aggregate = summary.get("aggregate", {})
    metrics = aggregate.get(mode)
    if metrics is None and aggregate:
        mode, metrics = next(iter(aggregate.items()))
    metrics = metrics or {}

    lines = [
        f"[ACoT-VLA] {label} completed",
        f"mode: {mode}",
        f"success: {_format_percent(metrics.get('success_rate'))}",
        f"timeout: {_format_percent(metrics.get('timeout_rate'))}",
        f"calls/episode: {_format_number(metrics.get('avg_deployable_policy_calls_per_episode'))}",
        f"avg H: {_format_number(metrics.get('avg_replan_horizon'))}",
        "deployable policy/episode: "
        + _format_seconds(metrics.get("avg_total_deployable_policy_inference_ms_per_episode")),
        "deployable wall/episode: "
        + _format_seconds(metrics.get("avg_total_deployable_wall_inference_ms_per_episode")),
        "observed wall/episode: " + _format_seconds(metrics.get("avg_total_wall_inference_ms_per_episode")),
    ]
    horizon_counts = metrics.get("execution_horizon_counts")
    if horizon_counts:
        lines.append(f"H distribution: {json.dumps(horizon_counts, sort_keys=True)}")

    per_task_path = output_dir / "per_task_summary.csv"
    if per_task_path.exists():
        lines.append("per-task success:")
        with per_task_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("mode") != mode:
                    continue
                lines.append(
                    f"  task {int(row['task_id'])}: {_format_percent(row.get('success_rate'))}, "
                    f"calls={_format_number(row.get('avg_deployable_policy_calls_per_episode'))}, "
                    f"H={_format_number(row.get('avg_replan_horizon'))}"
                )
    lines.append(f"output: {output_dir}")
    return "\n".join(lines)


def _progress_message(output_dir: pathlib.Path, *, label: str, pid: int) -> str:
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
            try:
                metrics = json.loads(lines[-1])
                return "\n".join(
                    [
                        f"[ACoT-VLA] {label} running",
                        f"pid: {pid}",
                        f"step: {metrics.get('step', 'n/a')}",
                        f"train loss: {_format_number(metrics.get('train/loss'), 4)}",
                        f"validation loss: {_format_number(metrics.get('validation/loss'), 4)}",
                        f"elapsed: {_format_number(metrics.get('elapsed_seconds'), 1)}s",
                        f"output: {output_dir}",
                    ]
                )
            except json.JSONDecodeError:
                pass
    log_path = output_dir / "run.log"
    tail = "run.log is not available"
    if log_path.exists():
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(log_lines[-5:])
    return (
        f"[ACoT-VLA] {label} running\n"
        f"pid: {pid}\n"
        f"output: {output_dir}\n"
        f"latest log:\n{tail}"
    )


def _failure_message(output_dir: pathlib.Path, *, label: str, pid: int) -> str:
    log_path = output_dir / "run.log"
    tail = "run.log is missing"
    if log_path.exists():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
    return (
        f"[ACoT-VLA] {label} stopped without summary.json\n"
        f"pid: {pid}\n"
        f"output: {output_dir}\n"
        f"last log lines:\n{tail}"
    )


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = _parse_args()
    webhook_url = os.environ.get(args.webhook_env)
    secret = os.environ.get(args.secret_env)
    if not webhook_url or not secret:
        raise RuntimeError(f"Missing {args.webhook_env} or {args.secret_env}.")

    if args.test_message is not None:
        _send_feishu(args.test_message, webhook_url=webhook_url, secret=secret)
        return

    output_dir = args.output_dir.resolve()
    marker_path = output_dir / ".feishu_notification_sent"
    if marker_path.exists():
        return

    next_progress = time.monotonic() + args.progress_seconds if args.progress_seconds else None
    while True:
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            message = _summary_message(output_dir, label=args.label, mode=args.mode)
            _send_feishu(message, webhook_url=webhook_url, secret=secret)
            marker_path.write_text("completed\n", encoding="utf-8")
            return

        if not _process_running(args.pid):
            time.sleep(5)
            if summary_path.exists():
                continue
            message = _failure_message(output_dir, label=args.label, pid=args.pid)
            _send_feishu(message, webhook_url=webhook_url, secret=secret)
            marker_path.write_text("failed\n", encoding="utf-8")
            return

        if next_progress is not None and time.monotonic() >= next_progress:
            message = _progress_message(output_dir, label=args.label, pid=args.pid)
            _send_feishu(message, webhook_url=webhook_url, secret=secret)
            next_progress = time.monotonic() + args.progress_seconds

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
