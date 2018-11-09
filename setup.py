from setuptools import setup

setup(
    name="gefolki",
    version="0.1",
    packages=["gefolki"],
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=["numpy", "scipy", "Pillow", "scikit-image"],
)
