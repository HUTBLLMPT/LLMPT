import os

import pkg_resources
from setuptools import setup, find_packages#定义项目的元数据和依赖关系

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),#自动找到并包含所有的包
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))#解析项目的依赖关系
        )
    ],
    include_package_data=True,#涵盖项目中的数据文件
    extras_require={'dev': ['pytest']},#开发时候用到，定义额外的依赖关系
)
