from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="transformer_inference_simulator",
    version="0.1.0",
    author="Dimitrios Kafetzis",
    author_email="dimitrioskafetzis@gmail.com",
    description="A simulator for distributed transformer inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dimitrios-Kafetzis/transformer_inference_simulator",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-experiments=run_experiments:main",
        ],
    },
)