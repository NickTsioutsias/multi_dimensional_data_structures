"""
FINAL DEMONSTRATION: Multi-dimensional Indexing with Text Similarity
This demonstrates the complete project as specified in the requirements
"""
import numpy as np
import time
from src.utils.data_loader import CoffeeDataLoader
from src.structures.kdtree import KDTree
from src.structures.quadtree import QuadTree
from src.structures.rangetree import RangeTree
from src.structures.rtree import SimpleRTree
from src.lsh.lsh import LSH
from src.hybrid_search import HybridSearch

print("="*70)
print("MULTI-DIMENSIONAL DATA STRUCTURES PROJECT")
print("FINAL DEMONSTRATION: SPATIAL + TEXT SIMILARITY SEARCH")
print("="*70)

# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================
print("\nPHASE 1: DATA PREPARATION")
print("-"*70)

# Load the full dataset (has review texts)
loader = CoffeeDataLoader('data/coffee_analysis.csv')
df = loader.preprocess()

# Create full review text
df['full_review'] = (
    df['desc_1'].fillna('') + ' ' +
    df['desc_2'].fillna('') + ' ' +
    df['desc_3'].fillna('')
).str.strip()

# Filter to get substantial reviews
df_full = df[df['full_review'].str.len() > 100].copy()
print(f"Loaded {len(df_full)} coffee reviews with text")

# Get spatial points and texts
points = df_full[['year', 'rating', '100g_USD', 'country_id']].values
texts = df_full['full_review'].tolist()

# Use first 1000 for performance
n_samples = min(1000, len(df_full))
points_subset = points[:n_samples]
texts_subset = texts[:n_samples]
df_subset = df_full.iloc[:n_samples].reset_index(drop=True)

print(f"Using {n_samples} samples for indexing")

# ============================================================================
# PHASE 2: BUILD ALL SPATIAL STRUCTURES
# ============================================================================
print("\nPHASE 2: BUILDING SPATIAL STRUCTURES")
print("-"*70)

structures = {}

# 1. K-D Tree
print("\n1. Building K-D Tree...")
start = time.time()
kd_tree = KDTree()
kd_tree.build(points_subset)
kd_time = time.time() - start
structures['K-D Tree'] = kd_tree
print(f"   Built in {kd_time*1000:.2f}ms")

# 2. Quad Tree
print("\n2. Building Quad Tree...")
start = time.time()
quad_tree = QuadTree(capacity=10)
quad_tree.build(points_subset)
quad_time = time.time() - start
structures['Quad Tree'] = quad_tree
print(f"   Built in {quad_time*1000:.2f}ms")

# 3. Range Tree
print("\n3. Building Range Tree...")
start = time.time()
range_tree = RangeTree(dimensions=4)
range_tree.build(points_subset)
range_time = time.time() - start
structures['Range Tree'] = range_tree
print(f"   Built in {range_time*1000:.2f}ms")

# 4. R-Tree (Simplified)
print("\n4. Building R-Tree...")
start = time.time()
r_tree = SimpleRTree()
r_tree.build(points_subset)
r_time = time.time() - start
structures['R-Tree'] = r_tree
print(f"   Built in {r_time*1000:.2f}ms")

# ============================================================================
# PHASE 3: BUILD LSH INDEX
# ============================================================================
print("\nPHASE 3: BUILDING LSH INDEX")
print("-"*70)

lsh = LSH(num_hashes=100, num_bands=10)
lsh.build(texts_subset)
print(f"LSH index built with {len(texts_subset)} documents")

# ============================================================================
# PHASE 4: HYBRID QUERY (PROJECT REQUIREMENT)
# ============================================================================
print("\nPHASE 4: HYBRID QUERY - SPATIAL + TEXT SIMILARITY")
print("-"*70)
print("\nProject Example Query:")
print("Find N-top most similar reviews from 2019-2021, rating > 94,")
print("price $4-$10, country=USA, similar to a specific review")

# Define spatial bounds
spatial_min = np.array([2019, 94, 4.0, 0])
spatial_max = np.array([2021, 100, 10.0, 50])  # Country 50 to include all

# Find a coffee with chocolate notes as our query
chocolate_idx = None
for i, text in enumerate(texts_subset[:100]):
    if 'chocolate' in text.lower():
        chocolate_idx = i
        break

if chocolate_idx:
    query_coffee = df_subset.iloc[chocolate_idx]
    print(f"\nQuery coffee: {query_coffee['name']}")
    print(f"Review snippet: {texts_subset[chocolate_idx][:150]}...")
else:
    chocolate_idx = 0
    print("Using first coffee as query")

# ============================================================================
# PHASE 5: COMPARE ALL STRUCTURES WITH HYBRID SEARCH
# ============================================================================
print("\nPHASE 5: PERFORMANCE COMPARISON")
print("-"*70)

results_summary = {}

for name, structure in structures.items():
    print(f"\n{name} + LSH:")
    print("-"*40)
    
    hybrid = HybridSearch(structure, lsh)
    results, stats = hybrid.hybrid_query(
        spatial_min, spatial_max,
        query_index=chocolate_idx,
        text_threshold=0.1,
        max_results=5
    )
    
    results_summary[name] = {
        'spatial_time': stats['spatial_time'],
        'lsh_time': stats['lsh_time'],
        'total_time': stats['total_time'],
        'spatial_results': stats['spatial_results'],
        'final_results': stats['final_results']
    }
    
    if results:
        print(f"\nTop 3 results:")
        for i, (point, idx, similarity) in enumerate(results[:3]):
            coffee = df_subset.iloc[idx]
            print(f"{i+1}. {coffee['name']} (similarity: {similarity:.1%})")

# ============================================================================
# PHASE 6: FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL PERFORMANCE SUMMARY")
print("="*70)

# Print comparison table
print(f"\n{'Structure':<15} | {'Build (ms)':<12} | {'Spatial (ms)':<12} | {'Total (ms)':<12} | {'Results':<10}")
print("-"*70)

build_times = {
    'K-D Tree': kd_time,
    'Quad Tree': quad_time,
    'Range Tree': range_time,
    'R-Tree': r_time
}

for name in structures.keys():
    stats = results_summary[name]
    print(f"{name:<15} | {build_times[name]*1000:<12.2f} | "
          f"{stats['spatial_time']*1000:<12.2f} | "
          f"{stats['total_time']*1000:<12.2f} | "
          f"{stats['final_results']:<10}")