from setuptools import setup, find_packages

__lib_name__ = "SpaDC"
__lib_version__ = "1.0.0"
__description__ = "Integrating spatial location, DNA sequence and chromatin accessibility via graph regularized convolutional neural network"
__url__ = "https://github.com/mcllllllll/SpaDC"
__author__ = "Chuanlong Ma"
__author_email__ = "2023202210190@whu.edu.cn"
__license__ = "MIT"
__keywords__ = ["spatial epigenomics"]
__requires__ = []

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['SpaDC'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)
