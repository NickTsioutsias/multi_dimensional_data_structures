"""
LSH (Locality Sensitive Hashing) implementation for text similarity
Using MinHash for document similarity
"""
import numpy as np
import time
from collections import defaultdict
import hashlib

class MinHash:
    """MinHash for estimating Jaccard similarity"""
    
    def __init__(self, num_hashes=100, seed=42):
        """
        Initialize MinHash
        
        Args:
            num_hashes: Number of hash functions to use
            seed: Random seed for reproducibility
        """
        self.num_hashes = num_hashes
        self.seed = seed
        np.random.seed(seed)
        
        # Generate random hash functions (a * x + b) % c
        self.hash_funcs = []
        max_hash = 2**32 - 1
        
        for _ in range(num_hashes):
            a = np.random.randint(1, max_hash)
            b = np.random.randint(0, max_hash)
            self.hash_funcs.append((a, b, max_hash))
    
    def create_shingles(self, text, k=3):
        """
        Create k-shingles from text
        
        Args:
            text: Input text
            k: Size of shingles (default 3-grams)
            
        Returns:
            Set of shingles
        """
        text = text.lower().strip()
        shingles = set()
        
        # Word-level shingles
        words = text.split()
        for i in range(len(words) - k + 1):
            shingle = ' '.join(words[i:i+k])
            shingles.add(shingle)
        
        # Also add character-level shingles for short texts
        if len(words) < k:
            for i in range(len(text) - k + 1):
                shingles.add(text[i:i+k])
        
        return shingles
    
    def compute_signature(self, shingles):
        """
        Compute MinHash signature for a set of shingles
        
        Args:
            shingles: Set of shingles
            
        Returns:
            MinHash signature (array of hash values)
        """
        if not shingles:
            return np.zeros(self.num_hashes)
        
        signature = np.full(self.num_hashes, np.inf)
        
        for shingle in shingles:
            # Hash the shingle to get an integer
            shingle_hash = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
            
            # Apply each hash function
            for i, (a, b, c) in enumerate(self.hash_funcs):
                hash_value = (a * shingle_hash + b) % c
                signature[i] = min(signature[i], hash_value)
        
        return signature
    
    def jaccard_similarity(self, sig1, sig2):
        """
        Estimate Jaccard similarity from signatures
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Estimated Jaccard similarity
        """
        return np.mean(sig1 == sig2)


class LSH:
    """LSH for finding similar documents"""
    
    def __init__(self, num_hashes=100, num_bands=20, seed=42):
        """
        Initialize LSH
        
        Args:
            num_hashes: Number of MinHash functions
            num_bands: Number of bands for LSH
            seed: Random seed
        """
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.minhash = MinHash(num_hashes, seed)
        
        # Storage
        self.signatures = []
        self.texts = []
        self.indices = []
        
        # Hash tables for each band
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
    
    def build(self, texts):
        """
        Build LSH index from texts
        
        Args:
            texts: List of text documents
        """
        print(f"Building LSH index with {len(texts)} documents...")
        start_time = time.time()
        
        self.texts = texts
        self.indices = list(range(len(texts)))
        
        # Compute signatures for all documents
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Processing document {i}/{len(texts)}...")
            
            # Create shingles and compute signature
            shingles = self.minhash.create_shingles(text)
            signature = self.minhash.compute_signature(shingles)
            self.signatures.append(signature)
            
            # Add to hash tables
            self._add_to_hash_tables(signature, i)
        
        build_time = time.time() - start_time
        print(f"LSH index built in {build_time:.3f} seconds")
    
    def _add_to_hash_tables(self, signature, doc_index):
        """Add a document to the LSH hash tables"""
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            # Get band signature
            band = tuple(signature[start_idx:end_idx])
            
            # Add to hash table
            self.hash_tables[band_idx][band].append(doc_index)
    
    def find_similar(self, query_text, threshold=0.5, max_results=10):
        """
        Find documents similar to query text
        
        Args:
            query_text: Query text
            threshold: Similarity threshold (0-1)
            max_results: Maximum number of results
            
        Returns:
            List of (index, similarity_score) tuples
        """
        start_time = time.time()
        
        # Compute signature for query
        query_shingles = self.minhash.create_shingles(query_text)
        query_signature = self.minhash.compute_signature(query_shingles)
        
        # Find candidate documents
        candidates = set()
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            band = tuple(query_signature[start_idx:end_idx])
            
            # Get all documents with the same band
            if band in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band])
        
        # Compute actual similarities for candidates
        results = []
        for candidate_idx in candidates:
            similarity = self.minhash.jaccard_similarity(
                query_signature, 
                self.signatures[candidate_idx]
            )
            
            if similarity >= threshold:
                results.append((candidate_idx, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        query_time = time.time() - start_time
        
        return results[:max_results], query_time
    
    def find_similar_by_index(self, doc_index, threshold=0.5, max_results=10):
        """
        Find documents similar to a document in the index
        
        Args:
            doc_index: Index of document in the collection
            threshold: Similarity threshold
            max_results: Maximum number of results
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if doc_index >= len(self.signatures):
            return [], 0
        
        query_signature = self.signatures[doc_index]
        
        start_time = time.time()
        
        # Find candidates
        candidates = set()
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            
            band = tuple(query_signature[start_idx:end_idx])
            
            if band in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band])
        
        # Remove the query document itself
        candidates.discard(doc_index)
        
        # Compute similarities
        results = []
        for candidate_idx in candidates:
            similarity = self.minhash.jaccard_similarity(
                query_signature,
                self.signatures[candidate_idx]
            )
            
            if similarity >= threshold:
                results.append((candidate_idx, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        query_time = time.time() - start_time
        
        return results[:max_results], query_time
    
    def get_stats(self):
        """Get LSH statistics"""
        total_buckets = sum(len(table) for table in self.hash_tables)
        avg_bucket_size = np.mean([
            len(bucket) for table in self.hash_tables 
            for bucket in table.values()
        ]) if total_buckets > 0 else 0
        
        return {
            'num_documents': len(self.texts),
            'num_hashes': self.num_hashes,
            'num_bands': self.num_bands,
            'rows_per_band': self.rows_per_band,
            'total_buckets': total_buckets,
            'avg_bucket_size': avg_bucket_size
        }