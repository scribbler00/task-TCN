from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="dies",
    version="0.1.0a0",
    description="Deep Intelligent Embedded Systems\n\nDeepLearning Models not only for Renewable Energy.",
    long_description=readme(),
    url="https://git.ies.uni-kassel.de/GettingDeep/dies",
    keywords="machine learning, energy timeseries, deep learning",
    author="Jens Schreiber, Janosch Henze",
    author_email="iescloud@uni-kassel.de",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    install_requires=[
        "numpy",
        "sklearn",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "fastai",
        "seaborn",
        "nose",
    ],
    test_suite="nose.collector",
    tests_require=["nose"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
    ],
)
