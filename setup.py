import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rlgym-analysis',
    version="0.1",
    author="Will Moschopoulos",
    description="A python API for analyzing RLGym rewards using data visualization and analysis techniques.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Enkhai/rlgym-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/Enkhai/rlgym-analysis/issues",
    },
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy',
        'pandas',
        'seaborn',
        'rlgym>=1.1.0'
    ]
)
