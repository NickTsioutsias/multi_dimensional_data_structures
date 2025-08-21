# Multi-Dimensional Data Structures with Text Similarity

Implementation and experimental evaluation of multi-dimensional data structures combined with LSH for text similarity search.

## Project Overview

This project implements and compares multiple spatial data structures for multi-dimensional indexing, combined with Locality Sensitive Hashing (LSH) for text similarity search. The system enables hybrid queries that filter by both spatial attributes and text content.

## Implemented Data Structures

### Spatial Structures
1. **K-D Tree**: Binary tree for k-dimensional data with O(log n) average query time
2. **Quad Tree**: Tree that recursively subdivides 2D space into quadrants
3. **Range Tree**: Optimized for orthogonal range queries with O(log^d n) query time
4. **R-Tree**: Uses Minimum Bounding Rectangles for spatial indexing

### Text Similarity
- **LSH with MinHash**: Locality Sensitive Hashing for efficient text similarity search
- Implements shingle generation, MinHash signatures, and band-based hashing

## Dataset

Using the Coffee Reviews Dataset from Kaggle with:
- 2000+ coffee reviews
- 4D spatial attributes: year, rating, price, country
- Text reviews with tasting notes and descriptions

## Key Features

- **Hybrid Search**: Combines spatial filtering with text similarity
- **Performance Comparison**: Benchmarks all structures on the same queries
- **Complete Operations**: Build, Insert, Delete, Range Query, kNN Query

## Performance Results

| Structure  | Build Time | Query Time | 
|------------|------------|------------|
| K-D Tree   | 11.77ms    | 0.57ms     |
| Quad Tree  | 9.76ms     | 0.39ms     |
| Range Tree | 2289.81ms  | 1.18ms     |
| R-Tree     | 0.15ms     | 0.86ms     |

## Example Query

"Find the top N most similar coffee reviews from 2019-2021, with rating > 94, price between $4-$10"

## Installation

```bash
pip install numpy pandas
```

## Usage

```bash
python final_demo.py
```

## Project Structure

```
src/
├── structures/      # Spatial data structures
├── lsh/            # LSH implementation
├── utils/          # Data loading utilities
└── hybrid_search.py # Hybrid query system
```
