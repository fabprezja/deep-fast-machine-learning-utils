from setuptools import setup, find_packages

setup(
    name="deepfastmlu",
    version="0.1",
    description="Machine learning utilities to help speed up your prototyping process.",
    author="Fabi Prezja",
    author_email="faprezja@fairn.fi",
    url="https://github.com/fabprezja/deep-fast-machine-learning-utils",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow>=2.3.0",
        "scikit-learn",
        "pandas",
        "seaborn",
        "matplotlib",
    ],
    extras_require={
        "gpu": ["tensorflow-gpu>=2.3.0"]
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)
