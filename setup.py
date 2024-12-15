from setuptools import setup, find_packages

setup(
    name="abm-lead-emergence",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pydantic"
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'coverage>=7.0.0',
        ],
        'dev': [
            'black>=22.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Agent-based modeling of leadership emergence",
    keywords="abm, leadership, emergence",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
) 