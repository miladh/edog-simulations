# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

install_requires = []

setup(
    name="edog",
    version=1.0,
    author='Milad H. Mobarhan',
    author_email='m@milad.no',
    license="GPLv3",
    packages=find_packages(),
    include_package_data=True,
)
