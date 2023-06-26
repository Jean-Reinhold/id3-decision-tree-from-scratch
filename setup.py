from setuptools import setup, find_packages

setup(
    name='id3-decision-tree',
    version='0.1',
    description='Implementation of ID3 algorithm for decision tree',
    author='Jean Reinhold',
    author_email='jeanpaulreinhold@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
    ],
)
