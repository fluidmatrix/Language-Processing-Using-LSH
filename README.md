# Approximate Nearest Neighbors using KNN & LSH

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![NLP](https://img.shields.io/badge/domain-NLP-orange) ![Algorithm](https://img.shields.io/badge/algorithm-KNN%20%7C%20LSH-purple) ![Status](https://img.shields.io/badge/status-educational-success)

## Table of Contents
* Project Overview
* Key Features
* Core Concepts
  * KNN
  * LSH
* Project Structure
* Code Workflow
* Requirements
* Applications
* References
* Author

---

## Project Overview
This repository demonstrates how **K-Nearest Neighbors (KNN)** can be scaled efficiently using **Locality Sensitive Hashing (LSH)** for high-dimensional natural language data.  
It applies these techniques to tweet embeddings to find semantically similar tweets.  
Traditional KNN is expensive for large datasets; LSH reduces the search space while preserving similarity.

---

## Key Features
* 📝 Tweet preprocessing & normalization
* 📄 Document embeddings using pretrained word vectors
* 🔍 Cosine similarity based KNN
* ⚡ Approximate nearest neighbor search using LSH
* 🌌 Multiple hash universes to improve recall

---

## Core Concepts

**K-Nearest Neighbors (KNN)**:
* Represents data points as vectors
* Computes similarity using cosine similarity
* Selects top *k* nearest neighbors
* Limitation: Brute-force KNN has O(N) per query time complexity.

**Locality Sensitive Hashing (LSH)**:
* Uses random hyperplanes to partition vector space
* Hashes similar vectors into the same buckets
* Restricts similarity search to a small subset of vectors
* Advantages:
  * Sub-linear query time
  * Efficient for high-dimensional data
  * Ideal for NLP & recommendation systems

---

## Project Structure
```python
.
├── helper.py        # NLP preprocessing, KNN & LSH utilities
├── main.py          # End-to-end execution pipeline
├── en_embeddings.p  # English word embeddings
├── fr_embeddings.p  # French word embeddings
├── README.md        # Documentation
```

## 🧹 TWEET PREPROCESSING
processed_tweet = process_tweet(tweet)  # Remove URLs, hashtags, stopwords, tokenize, stem

## 🧩 DOCUMENT EMBEDDINGS
doc_embedding = get_document_embedding(tweet, en_embeddings)
document_vec_matrix, ind2Tweet_dict = get_document_vecs(all_tweets, en_embeddings)

## 🔑 LSH HASH TABLES
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES)) for _ in range(N_UNIVERSES)]
hash_tables, id_tables = create_hash_id_tables(N_UNIVERSES)

## ⚡ APPROXIMATE KNN SEARCH
nearest_neighbor_ids = approximate_knn(
    doc_id, vec_to_search, planes_l, hash_tables, id_tables, k=3, num_universes_to_use=5
)

## 📢 Display Results
print(f"Nearest neighbors for document {doc_id}:")
for neighbor_id in nearest_neighbor_ids:
    print(ind2Tweet_dict[neighbor_id])

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
