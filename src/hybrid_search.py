"""
Hybrid search combining spatial indexing with LSH text similarity
This is the main goal of the project!
"""
import numpy as np
import time
from typing import List, Tuple, Dict

class HybridSearch:
    """
    Combines spatial data structures with LSH for hybrid queries
    Example: Find coffee reviews from 2019-2021, rating > 94, price $4-10, 
             that are similar to a given review text
    """
    
    def __init__(self, spatial_structure, lsh_index):
        """
        Initialize hybrid search
        
        Args:
            spatial_structure: One of KDTree, QuadTree, RangeTree, or RTree
            lsh_index: LSH index for text similarity
        """
        self.spatial = spatial_structure
        self.lsh = lsh_index
        self.structure_name = spatial_structure.__class__.__name__
        
    def hybrid_query(self, 
                    spatial_min: np.ndarray,
                    spatial_max: np.ndarray,
                    query_text: str = None,
                    query_index: int = None,
                    text_threshold: float = 0.15,
                    max_results: int = 10) -> Tuple[List, Dict]:
        """
        Perform hybrid query: spatial filtering + text similarity
        
        Args:
            spatial_min: Min bounds for spatial query [year, rating, price, country]
            spatial_max: Max bounds for spatial query
            query_text: Text to find similar reviews (if provided)
            query_index: Index of document to find similar (if provided)
            text_threshold: Minimum text similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (results, statistics)
        """
        stats = {
            'spatial_structure': self.structure_name,
            'spatial_time': 0,
            'lsh_time': 0,
            'total_time': 0,
            'spatial_results': 0,
            'lsh_candidates': 0,
            'final_results': 0
        }
        
        total_start = time.time()
        
        # Step 1: Spatial filtering
        print(f"\n1. Spatial Query using {self.structure_name}")
        print(f"   Range: Year {spatial_min[0]:.0f}-{spatial_max[0]:.0f}, "
              f"Rating {spatial_min[1]:.0f}-{spatial_max[1]:.0f}, "
              f"Price ${spatial_min[2]:.1f}-${spatial_max[2]:.1f}")
        
        spatial_start = time.time()
        
        # Call appropriate method based on structure type
        if hasattr(self.spatial, 'range_query'):
            spatial_results, spatial_time = self.spatial.range_query(spatial_min, spatial_max)
        else:
            # For structures that don't return time
            spatial_results = self.spatial.range_query(spatial_min, spatial_max)
            spatial_time = time.time() - spatial_start
        
        stats['spatial_time'] = spatial_time
        stats['spatial_results'] = len(spatial_results)
        
        print(f"   Found {len(spatial_results)} coffees in {spatial_time*1000:.2f}ms")
        
        # If no text query, return spatial results
        if query_text is None and query_index is None:
            stats['final_results'] = len(spatial_results)
            stats['total_time'] = time.time() - total_start
            return spatial_results[:max_results], stats
        
        # Step 2: Text similarity filtering
        print(f"\n2. Text Similarity using LSH")
        
        lsh_start = time.time()
        
        # Get text similarities
        if query_text is not None:
            print(f"   Query text: '{query_text[:100]}...'")
            text_results, _ = self.lsh.find_similar(query_text, 
                                                   threshold=text_threshold, 
                                                   max_results=len(self.lsh.texts))
        else:
            print(f"   Finding similar to document {query_index}")
            text_results, _ = self.lsh.find_similar_by_index(query_index,
                                                            threshold=text_threshold,
                                                            max_results=len(self.lsh.texts))
        
        text_indices = set(idx for idx, _ in text_results)
        stats['lsh_time'] = time.time() - lsh_start
        stats['lsh_candidates'] = len(text_results)
        
        print(f"   Found {len(text_results)} similar texts in {stats['lsh_time']*1000:.2f}ms")
        
        # Step 3: Intersect results
        print(f"\n3. Combining Results")
        print(f"   Spatial results count: {len(spatial_results)}")
        print(f"   LSH candidates count: {len(text_indices)}")
        
        final_results = []
        spatial_indices = [idx for _, idx in spatial_results]
        
        for point, idx in spatial_results:
            if idx in text_indices:
                # Find similarity score
                similarity = next((sim for text_idx, sim in text_results if text_idx == idx), 0)
                final_results.append((point, idx, similarity))
        
        # Sort by similarity score
        final_results.sort(key=lambda x: x[2], reverse=True)
        
        stats['final_results'] = len(final_results)
        stats['total_time'] = time.time() - total_start
        
        print(f"   Final: {len(final_results)} coffees match both criteria")
        print(f"   Total query time: {stats['total_time']*1000:.2f}ms")
        
        return final_results[:max_results], stats
    
    def hybrid_query_ranked(self,
                           spatial_min: np.ndarray,
                           spatial_max: np.ndarray,
                           query_index: int = None,
                           max_results: int = 10) -> Tuple[List, Dict]:
        """
        Alternative approach: Return spatial results ranked by text similarity
        (Even if similarity is low)
        
        Args:
            spatial_min: Min bounds for spatial query
            spatial_max: Max bounds for spatial query
            query_index: Index of document for similarity
            max_results: Maximum number of results
            
        Returns:
            Tuple of (results, statistics)
        """
        stats = {
            'spatial_structure': self.structure_name,
            'spatial_time': 0,
            'ranking_time': 0,
            'total_time': 0,
            'spatial_results': 0,
            'final_results': 0
        }
        
        total_start = time.time()
        
        # Get spatial results
        print(f"\n1. Spatial Query using {self.structure_name}")
        spatial_start = time.time()
        
        if hasattr(self.spatial, 'range_query'):
            spatial_results, spatial_time = self.spatial.range_query(spatial_min, spatial_max)
        else:
            spatial_results = self.spatial.range_query(spatial_min, spatial_max)
            spatial_time = time.time() - spatial_start
        
        stats['spatial_time'] = spatial_time
        stats['spatial_results'] = len(spatial_results)
        print(f"   Found {len(spatial_results)} coffees")
        
        if query_index is None:
            # No ranking, just return spatial results
            stats['total_time'] = time.time() - total_start
            stats['final_results'] = min(len(spatial_results), max_results)
            return spatial_results[:max_results], stats
        
        # Rank by text similarity
        print(f"\n2. Ranking by text similarity to document {query_index}")
        ranking_start = time.time()
        
        query_signature = self.lsh.signatures[query_index]
        results_with_similarity = []
        
        for point, idx in spatial_results:
            if idx < len(self.lsh.signatures):
                similarity = self.lsh.minhash.jaccard_similarity(
                    query_signature,
                    self.lsh.signatures[idx]
                )
                results_with_similarity.append((point, idx, similarity))
        
        # Sort by similarity
        results_with_similarity.sort(key=lambda x: x[2], reverse=True)
        
        stats['ranking_time'] = time.time() - ranking_start
        stats['final_results'] = min(len(results_with_similarity), max_results)
        stats['total_time'] = time.time() - total_start
        
        print(f"   Ranked {len(results_with_similarity)} results")
        print(f"   Top similarity: {results_with_similarity[0][2]:.1%}" if results_with_similarity else "   No results")
        
        return results_with_similarity[:max_results], stats
    
    def compare_structures(self, structures_dict: Dict,
                          spatial_min: np.ndarray,
                          spatial_max: np.ndarray,
                          query_text: str = None,
                          text_threshold: float = 0.15) -> Dict:
        """
        Compare performance of different spatial structures for hybrid query
        
        Args:
            structures_dict: Dictionary of {name: structure} pairs
            spatial_min: Min bounds for spatial query
            spatial_max: Max bounds for spatial query
            query_text: Text query
            text_threshold: Text similarity threshold
            
        Returns:
            Dictionary of performance statistics
        """
        comparison = {}
        
        for name, structure in structures_dict.items():
            print(f"\n{'='*50}")
            print(f"Testing {name}")
            print('='*50)
            
            hybrid = HybridSearch(structure, self.lsh)
            results, stats = hybrid.hybrid_query(
                spatial_min, spatial_max,
                query_text=query_text,
                text_threshold=text_threshold
            )
            
            comparison[name] = {
                'stats': stats,
                'results': results
            }
        
        return comparison