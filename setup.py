from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='Loan-Default-Prediction-alanchn31',
    version='0.1dev',
    author="Alan Choon",
    author_email="alanchn31@gmail.com",
    description="A small example package showing pyspark model training",
    long_description=long_description,
    packages=setuptools.find_packages(),
    license='''
    Creative Commons
    Attribution-Noncommercial-Share Alike license''',
    python_requires='>=3.6',
)