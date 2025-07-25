#!/bin/bash
# Setup script for RAG-Lite environment on Linux/macOS

echo "Creating conda environment for RAG-Lite..."
conda create -n raglite python=3.10 -y

echo "Activating raglite environment..."
source conda activate raglite

echo "Installing dependencies..."
pip install sentence-transformers==2.7.0 faiss-cpu==1.7.2 \
     transformers==4.39.3 peft==0.10.0 \
     datasets==2.18.0 fastapi==0.110 uvicorn==0.29 \
     torch accelerate bitsandbytes \
     numpy pandas nltk beautifulsoup4 python-dotenv tensorboard

echo "Environment setup complete!"
echo "To activate the environment: conda activate raglite"
