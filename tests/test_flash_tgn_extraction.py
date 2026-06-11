import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


def test_clean_package_uses_local_modules():
    import flash_tgn.autograd_ops as autograd_ops
    import flash_tgn.model as model
    import flash_tgn.schedule as schedule
    import flash_tgn.temporal_index as temporal_index
    import flash_tgn.trainer as trainer

    modules = [autograd_ops, model, schedule, temporal_index, trainer]
    for module in modules:
        assert '/python/flash_tgn/' in (module.__file__ or '')


def test_clean_extension_loader_targets_flash_namespace():
    from flash_tgn.extension import cuda_extension

    try:
        ext = cuda_extension()
    except ModuleNotFoundError as exc:
        assert 'flash_tgn._C' in str(exc)
    else:
        assert ext.__name__ == 'flash_tgn._C'
