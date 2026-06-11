"""Policy helpers for choosing a safe epoch precompute mode."""

from __future__ import annotations


GPU_PRECOMP_LIMIT_BYTES = 10_000_000_000
CPU_PRECOMP_LIMIT_BYTES = 80_000_000_000
HOST_PINNED_BUDGET_FRACTION = 0.30
CPU_PINNED_OVERHEAD_FACTOR_1L = 2.2
ONLINE_PRECOMP_CHUNK_BATCHES = 32


def get_host_mem_available_bytes() -> int | None:
    """Return MemAvailable from /proc/meminfo if available."""
    try:
        with open('/proc/meminfo', 'r', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def choose_epoch_precompute_mode(
    precomp_bytes_est: int,
    n_layers: int,
    host_mem_available: int | None,
    gpu_precomp_limit: int = GPU_PRECOMP_LIMIT_BYTES,
    cpu_precomp_limit: int = CPU_PRECOMP_LIMIT_BYTES,
    host_budget_fraction: float = HOST_PINNED_BUDGET_FRACTION,
    cpu_overhead_factor_1layer: float = CPU_PINNED_OVERHEAD_FACTOR_1L,
) -> tuple[str, str]:
    """Choose between GPU precompute, CPU-pinned precompute, and online sampling."""
    if precomp_bytes_est < gpu_precomp_limit:
        return 'gpu', 'fits_gpu_budget'

    if precomp_bytes_est >= cpu_precomp_limit:
        return 'online', 'exceeds_host_precompute_limit'

    if n_layers >= 2:
        return 'online', 'two_layer_host_precompute_disabled'

    if host_mem_available is None:
        return 'online', 'host_memory_unknown'

    host_precomp_est = int(precomp_bytes_est * cpu_overhead_factor_1layer)
    host_budget = int(host_mem_available * host_budget_fraction)
    if host_precomp_est <= host_budget:
        return 'cpu', 'fits_host_budget'
    return 'online', 'insufficient_host_budget'


def should_build_full_epoch_precompute(precomp, precomp_mode: str) -> bool:
    """Return whether run_epoch should materialize the whole epoch at once."""
    return precomp is None and precomp_mode != 'online'

