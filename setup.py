from setuptools import find_packages, setup

setup(
    name="crossmodal",
    version="0.0",
    description="Crossmodal filtering",
    url="http://github.com/njanne19/kittifilter",
    author="njanne19",
    author_email="njanne@umich.edu",
    license="BSD",
    packages=["crossmodal"],
    install_requires=[
        "fannypack",
        "torchfilter",
    ],
)
