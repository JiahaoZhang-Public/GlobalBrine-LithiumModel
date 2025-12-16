from __future__ import annotations

import os
from typing import Any, Optional


class WandbUnavailableError(RuntimeError):
    pass


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: Optional[str],
    mode: str,
    config: dict[str, Any],
    tags: tuple[str, ...],
) -> Any:
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise WandbUnavailableError(
            "wandb is not installed. Install it (e.g. `pip install wandb`) "
            "or run with --no-wandb."
        ) from exc

    if mode:
        os.environ.setdefault("WANDB_MODE", mode)

    return wandb.init(
        project=project,
        name=run_name,
        config=config,
        tags=list(tags) if tags else None,
    )


def log_wandb(run: Any, metrics: dict[str, Any], *, step: Optional[int] = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)
