"""Set up package."""
import setuptools

REQUIRES = [
        'numpy', 'tensorflow'
]


setuptools.setup(
    name="sllutils",
    version="0.1.0",
    url="https://github.com/cwinsnes/sllutils",
    author="Casper Winsnes",
    author_email="cwinsnes92@gmail.com",
    description="Code snippets that I at some point or another thought useful",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=REQUIRES,
    entry_points={
        "console_scripts": []},
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
