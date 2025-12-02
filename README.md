# Retrieval-Augmented Generation (RAG) Learning Projects

This repository is a collection of small, focused projects designed to build a practical understanding of Retrieval-Augmented Generation systems. Each project isolates one concept, implements it end-to-end, and demonstrates how it contributes to a reliable RAG pipeline.

The goal is to break down the full RAG workflow into manageable pieces — from preprocessing and chunking, to vector storage, retrieval strategies, model selection, evaluation, and production deployment. Instead of large, abstract examples, every module here is built to be concrete, minimal, and directly applicable to real work.

## What This Repository Covers

The material here spans the entire lifecycle of a RAG system:

- Working with raw text and preparing it for retrieval  
- Different strategies for chunking and why they matter  
- Building and tuning vector indexes  
- Comparing embedding models and understanding their limitations  
- Retrieval patterns such as semantic search, hybrid search, and re-ranking  
- Structuring multi-document workflows  
- Keeping responses grounded in source material  
- Improving reliability with corrective or query-expansion techniques  
- Adding images, code, structured formats, or other modalities  
- Exposing a RAG pipeline as an API  
- Managing cost, performance, caching, and monitoring in real deployments  

Every project is small by design. The emphasis is on clarity and practical insight rather than large frameworks or abstract theory.

## How to Use This Repository

Each folder contains:
- A focused topic
- A short explanation of the concept
- One or more scripts that implement it
- Example inputs and expected outputs

You can explore in any order, but the concepts build on one another naturally. Starting with text processing and vector storage will make later topics like retrieval optimization and evaluation much easier to understand.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
