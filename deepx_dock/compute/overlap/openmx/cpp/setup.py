"""
Build script for overlap_openmx C++ extension.

Usage:
    python setup.py build_ext --inplace

Requirements:
    - CMake >= 3.15
    - Eigen3 >= 3.3
    - HDF5 with C++ support
    - pybind11
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
from pathlib import Path


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = Path().absolute()

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={self.python_executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = [
            "--config",
            "Release",
            "-j",
            "4",
        ]

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)

        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


setup(
    name="overlap_openmx",
    version="0.1.0",
    description="OpenMX-style overlap matrix calculation",
    author="DeepH Team",
    ext_modules=[CMakeExtension("overlap_openmx", ".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
)
