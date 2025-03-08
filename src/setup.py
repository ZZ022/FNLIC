from setuptools import setup, find_packages
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = dict(
    rans=Pybind11Extension(
        name=f"cbench.rans",
        sources=["cbench/csrc/rans/rans_interface.cpp",],
        language="c++",
        include_dirs=["cbench/csrc/rans"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
    ),
    ar=Pybind11Extension(
        name=f"cbench.ar",
        sources=glob("cbench/csrc/ar/*.cpp"),
        language="c++",
        include_dirs=["cbench/csrc/ar"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
    ),
    ans=Pybind11Extension(
        name=f"cbench.ans",
        sources=glob("cbench/csrc/ans/*.cpp"),
        language="c++",
        include_dirs=["cbench/csrc/ans"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
    ),
)

cmdclass = {
    'build_ext': build_ext,
}

# skip check requires to speed up installation, but may be needed for production
install_requires = [
    # # basic
    # "numpy",
    # "scipy",
    # # pytorch
    # "torch",
    # "torchvision",
    # "pytorch-lightning",
    # # compression libs
    # "zstandard",
    # "brotlipy",
    # # other
    # # "cython", # cython needed to be installed beforehand for pandas
    # "tqdm",
    # "pandas",
]

def generate_pyi(module_name):
    import torch
    from pybind11_stubgen import ModuleStubsGenerator

    module = ModuleStubsGenerator("cbench."+module_name)
    module.parse()
    module.write_setup_py = False

    with open("cbench/%s.pyi" % module_name, "w") as fp:
        fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

# TODO: 'CC=clang CXX=clang++' on macos ?
setup(
    name="cbench",
    version="0.1",
    description="Data Compression Benchmark",
    url="",
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=list(ext_modules.values()),
    cmdclass=cmdclass,
)


# test if the extension is built correctly
import importlib
import traceback
for ext in ext_modules.values():
    print(f"Try import {ext.name}")
    try:
        importlib.import_module(ext.name)
    except ImportError as e:
        print(f"Import {ext.name} failed!")
        traceback.print_exc()
        exit(1)
    print(f"Import {ext.name} success!")
print("Setup success!")

for module_name in ext_modules.keys():
    generate_pyi(module_name)