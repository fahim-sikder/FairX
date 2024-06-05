from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(f"{here}/requirements.txt") as f:
    install_requires_pack = [line.strip() for line in f]

VERSION = '0.0.1'
DESCRIPTION = 'Fairness Evaluation Suite'
LONG_DESCRIPTION = 'A Python package to assess and improve fairness of machine learning models.'

# Setting up
setup(
    name="fairx",
    version=VERSION,
    author="Md Fahim Sikder",
    author_email="<fahimsikder01@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires_pack,
    keywords=['python', 'fair data', 'fairness', 'evaluation', 'doublex'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)