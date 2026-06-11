import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cuda_arch = os.environ.get('FLASH_TGN_CUDA_ARCH', 'sm_90')

CUDA_SOURCE_FILES = [
    'bind.cpp',
    'tt_csr.cu',
    'fused_gather.cu',
    'fused_gather_v2.cu',
    'fused_lse_gather.cu',
    'fused_index_scatter.cu',
    'fused_l0_gather.cu',
    'fused_attn.cu',
]

FLASH_CUDA_DIR = 'csrc_flash_tgn'

FLASH_CUDA_SOURCES = [
    os.path.join(FLASH_CUDA_DIR, source) for source in CUDA_SOURCE_FILES
]

COMPILE_ARGS = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3', '-std=c++17',
        f'-arch={cuda_arch}',
        '--use_fast_math',
        '-Xptxas=-v',
    ],
}

setup(
    name='flash-tgn',
    ext_modules=[
        CUDAExtension(
            name='flash_tgn._C',
            sources=FLASH_CUDA_SOURCES,
            extra_compile_args=COMPILE_ARGS,
            include_dirs=[FLASH_CUDA_DIR],
        ),
    ],
    packages=find_packages('python'),
    package_dir={'': 'python'},
    cmdclass={'build_ext': BuildExtension},
)
