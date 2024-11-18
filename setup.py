from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="brainglobe_ccf_translator",
    version="0.21",
    packages=find_packages(),
    license="MIT",
    description="a package to translate data between common coordinate frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={"brainglobe_ccf_translator": ["metadata/translation_metadata.csv"]},
    include_package_data=True,
    install_requires=["numpy", "nibabel", "scipy", "networkx", "pandas", "requests"],
    url='https://github.com/brainglobe/brainglobe-ccf-translator',
    author_email='harry.carey@medisin.uio.no',
    keywords=['neuroinformatics', 'brain', 'atlas', 'brainglobe', 'neuroscience'],
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
],
)