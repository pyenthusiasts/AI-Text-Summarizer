"""Setup script for AI Text Summarizer."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()
dev_requirements = [
    r.strip()
    for r in dev_requirements
    if r.strip() and not r.startswith("#") and not r.startswith("-r")
]

setup(
    name="ai-text-summarizer",
    version="1.0.0",
    author="Python Enthusiasts",
    description="A Python package for text summarization using Hugging Face Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyenthusiasts/AI-Text-Summarizer",
    project_urls={
        "Bug Reports": "https://github.com/pyenthusiasts/AI-Text-Summarizer/issues",
        "Source": "https://github.com/pyenthusiasts/AI-Text-Summarizer",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "ai-text-summarizer=ai_text_summarizer.cli:main",
        ],
    },
    keywords="text summarization nlp transformers huggingface ai machine-learning",
    include_package_data=True,
)
