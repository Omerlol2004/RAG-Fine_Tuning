"""
Setup script for the RAG-Lite package.
"""

from setuptools import setup, find_packages

setup(
    name="rag_lite",
    version="0.1.0",
    description="A lightweight Retrieval-Augmented Generation implementation",
    author="RAG-Lite Team",
    author_email="example@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "transformers==4.39.3",
        "peft==0.10.0",
        "datasets==2.18.0",
        "sentence-transformers==2.7.0",
        "faiss-cpu==1.7.2",
        "torch>=2.0.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.0",
        "nltk>=3.8.0",
        "beautifulsoup4>=4.12.0",
        "fastapi==0.110",
        "uvicorn==0.29",
        "python-dotenv>=1.0.0",
        "tensorboard>=2.14.0",
    ],
    entry_points={
        "console_scripts": [
            "rag-lite=rag_lite.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

