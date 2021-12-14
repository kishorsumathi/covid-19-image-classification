from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="kishor",
    description="A small package for covid-19 detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kishorsumathi/covid-19-image-classification.git",
    author_email="kishorbrindha18@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "dvc",
        "tensorflow",
        "matplotlib",
        "numpy",
        "pandas",
        "split-folders",
        "PyYAML",
    ]
)