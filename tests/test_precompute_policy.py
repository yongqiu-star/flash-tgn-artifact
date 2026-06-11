import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from flash_tgn.precompute_policy import (
    choose_epoch_precompute_mode,
    should_build_full_epoch_precompute,
)


def test_small_precompute_stays_on_gpu():
    mode, reason = choose_epoch_precompute_mode(
        precomp_bytes_est=8_000_000_000,
        n_layers=2,
        host_mem_available=200_000_000_000,
    )
    assert mode == 'gpu'
    assert reason == 'fits_gpu_budget'


def test_medium_1layer_can_use_cpu_pinned():
    mode, reason = choose_epoch_precompute_mode(
        precomp_bytes_est=16_000_000_000,
        n_layers=1,
        host_mem_available=200_000_000_000,
    )
    assert mode == 'cpu'
    assert reason == 'fits_host_budget'


def test_medium_2layer_forces_online_mode():
    mode, reason = choose_epoch_precompute_mode(
        precomp_bytes_est=16_000_000_000,
        n_layers=2,
        host_mem_available=200_000_000_000,
    )
    assert mode == 'online'
    assert reason == 'two_layer_host_precompute_disabled'


def test_medium_1layer_falls_back_when_host_budget_is_tight():
    mode, reason = choose_epoch_precompute_mode(
        precomp_bytes_est=16_000_000_000,
        n_layers=1,
        host_mem_available=20_000_000_000,
    )
    assert mode == 'online'
    assert reason == 'insufficient_host_budget'


def test_online_mode_skips_full_epoch_precompute():
    assert not should_build_full_epoch_precompute(None, precomp_mode='online')


def test_epoch_mode_without_precomp_builds_full_epoch_precompute():
    assert should_build_full_epoch_precompute(None, precomp_mode='gpu')
