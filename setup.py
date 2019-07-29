import os

from pip._internal.req import parse_requirements
from setuptools import setup, find_packages

print (os.listdir("./"))

requirements = parse_requirements('./requirements.txt', session='hack')
requirements = [str(ir.req) for ir in requirements]
print (requirements)
setup(
    name='Distinctiopus4',
    packages=find_packages(),
    url='https://github.com/c0ntradicti0n/Distinctiopus4.git',
    description='This is NLP software to mine propositions of difference explanations (in texts).',
    long_description=open('./README.md').read(),
    install_requires= requirements,
    dependency_links = [
     "git+git://github.com/c0ntradicti0n/Distinctiopus4/tarball/master#egg=Distinctiopus4",
    ],
    include_package_data=True,
)