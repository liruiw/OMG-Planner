from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="omg",
    ext_modules=[
        CUDAExtension(
            name="omg_cuda",
            sources=["sdf_matching_loss_kernel.cu", "omg_layers.cpp"],
            include_dirs=["/usr/local/include/eigen3", "/usr/local/include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
