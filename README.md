Multi-Dimensional Data Structures with Text Similarity
Implementation and experimental evaluation of multi-dimensional data structures combined with LSH for text similarity search.
Project Overview
This project implements and compares multiple spatial data structures for multi-dimensional indexing, combined with Locality Sensitive Hashing (LSH) for text similarity search. The system enables hybrid queries that filter by both spatial attributes and text content.
Implemented Data Structures
Spatial Structures

K-D Tree: Binary tree for k-dimensional data with O(log n) average query time
Quad Tree: Tree that recursively subdivides 2D space into quadrants
Range Tree: Optimized for orthogonal range queries with O(log^d n) query time
R-Tree: Uses Minimum Bounding Rectangles for spatial indexing

Text Similarity

LSH with MinHash: Locality Sensitive Hashing for efficient text similarity search
Implements shingle generation, MinHash signatures, and band-based hashing

Dataset
Using the Coffee Reviews Dataset from Kaggle with:

2000+ coffee reviews
4D spatial attributes: year, rating, price, country
Text reviews with tasting notes and descriptions

Key Features

Hybrid Search: Combines spatial filtering with text similarity
Performance Comparison: Benchmarks all structures on the same queries
Complete Operations: Build, Insert, Delete, Range Query, kNN Query
