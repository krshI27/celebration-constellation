"""Setup configuration for celebration-constellation package."""

from setuptools import find_packages, setup

setup(
    name="celebration-constellation",
    version="0.1.0",
    author="Maximilian Sperlich",
    description="Star constellation matching from table photos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/krshI27/celebration-constellation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0",
        "opencv-python-headless>=4.0",
        "scikit-image>=0.25",
        "astropy>=7.0",
        "astroquery>=0.4",
        "scipy>=1.16",
        "streamlit>=1.51",
        "pandas>=2.0",
        "pillow>=12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0",
            "pytest-cov>=7.0",
            "black>=25.0",
            "flake8>=7.0",
            "python-dotenv>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
