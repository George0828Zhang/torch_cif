import os
from setuptools import find_packages, setup


base_dir = os.path.dirname(os.path.abspath(__file__))


def get_version():
    version_path = os.path.join(base_dir, "torch_cif", "__init__.py")
    version = {}
    with open(version_path, encoding="utf-8") as fp:
        exec(fp.read(), version)
    return version["__version__"]


setup(
    name='torch_cif',
    packages=find_packages(),
    version=get_version(),
    license='MIT',
    description='A fast parallel implementation of continuous'
    ' integrate-and-fire (CIF) https://arxiv.org/abs/1905.11235',
    author='Chih-Chiang Chang',
    author_email='cc.chang0828@gmail.com',
    url='https://github.com/George0828Zhang/torch_cif',
    keywords=" ".join([
        'speech', 'speech-recognition', 'asr', 'automatic-speech-recognition',
        'speech-to-text', 'speech-translation',
        'continuous-integrate-and-fire', 'cif',
        'monotonic', 'alignment', 'torch', 'pytorch'
    ]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=["torch"],
    extras_require={
        "test": [
            "hypothesis",
            "expecttest"
        ],
    },
    include_package_data=True,
)
