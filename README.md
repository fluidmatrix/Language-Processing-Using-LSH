# Approximate Nearest Neighbors using KNN and Locality Sensitive Hashing

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![NLP](https://img.shields.io/badge/domain-NLP-orange)
![Algorithm](https://img.shields.io/badge/algorithm-KNN%20%7C%20LSH-purple)
![Status](https://img.shields.io/badge/status-educational-success)

## Table of Contents
* Project Overview
* Installation
* Usage
* Algorithms
* Results

# Approximate Nearest Neighbors using KNN and Locality Sensitive Hashing

This repository demonstrates how **K-Nearest Neighbors (KNN)** can be scaled efficiently using **Locality Sensitive Hashing (LSH)** for high-dimensional natural language data.  
The project applies these techniques to tweet embeddings to find semantically similar tweets.

---

## Project Overview

Traditional KNN requires comparing a query vector with every vector in the dataset, which becomes computationally expensive for large datasets.  
This project uses Locality Sensitive Hashing to reduce the search space while preserving similarity.

---

## Key Features

* Tweet preprocessing and normalization
* Document embeddings using word vectors
* Cosine similarity based KNN
* Approximate nearest neighbor search using LSH
* Multiple hash universes to improve recall

---

## Core Concepts

### K-Nearest Neighbors (KNN)

KNN is a distance-based algorithm that:

* Represents data points as vectors
* Computes similarity using cosine similarity
* Selects the top *k* nearest neighbors

**Limitation:**  
Brute-force KNN has a time complexity of *O(N)* per query.

---

### Locality Sensitive Hashing (LSH)

Locality Sensitive Hashing is an approximate nearest neighbor technique that:

* Uses random hyperplanes to partition vector space
* Hashes similar vectors into the same buckets
* Restricts similarity search to a small subset of vectors

**Advantages:**

* Sub-linear query time
* Efficient for high-dimensional data
* Suitable for NLP and recommendation systems

---

## Project Structure
.
├── helper.py # NLP preprocessing, KNN, and LSH utilities
├── main.py # End-to-end execution pipeline
├── en_embeddings.p # English word embeddings
├── fr_embeddings.p # French word embeddings
├── README.md # Documentation


---

## Code Workflow

### Tweet Preprocessing

Tweets are cleaned by:

* Removing URLs, hashtags, and stopwords
* Tokenizing text
* Applying stemming

process_tweet(tweet)

---

### Document Embeddings

Each tweet is converted into a 300-dimensional vector by summing word embeddings.

## Requirements

This project uses a **frozen dependency setup** to ensure reproducibility across environments.

All required packages and their exact versions are listed in `requirements.txt`.

### Python Version

* Python **3.9+** (recommended)

---

### Install Dependencies
```python
pip install -r requirements.txt
```

### Import nltk Data
```python
import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
```

---

## Applications

* Tweet similarity and clustering
* Semantic search
* Recommendation systems
* Large-scale NLP systems

---

## References

* Mining of Massive Datasets – Jure Leskovec
* Locality Sensitive Hashing
* Cosine Similarity in NLP
* NLTK Twitter Samples Dataset
* This is working implementation of DeepLearning.AI NLP course

---

## Author

This project demonstrates efficient scaling of KNN using Locality Sensitive Hashing for NLP tasks.
