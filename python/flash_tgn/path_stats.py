from __future__ import annotations


def reset_attention_path_counts() -> None:
    from .operator_policy import reset_operator_counters

    reset_operator_counters()


def format_attention_path_counts() -> str | None:
    from .operator_policy import format_operator_counters

    return format_operator_counters()
