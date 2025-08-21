"""
Fixed version to show actual results in hybrid search
"""
import numpy as np
from src.utils.data_loader import CoffeeDataLoader
from src.structures.kdtree import KDTree
from src.lsh.lsh import LSH
from src.hybrid_search import HybridSearch

print("="*70)
print("HYBRID SEARCH - FIXED DEMONSTRATION")
print("="*70)

# Load data
loader = CoffeeDataLoader('data/coffee_analysis.csv')
df = loader.preprocess()

# Create review texts
df['full_review'] = (
    df['desc_1'].fillna('') + ' ' +
    df['desc_2'].fillna('') + ' ' +
    df['desc_3'].fillna('')
).str.strip()

df_full = df[df['full_review'].str.len() > 100].copy()
print(f"Loaded {len(df_full)} coffee reviews")

# Get data
points = df_full[['year', 'rating', '100g_USD', 'country_id']].values[:1000]
texts = df_full['full_review'].tolist()[:1000]
df_subset = df_full.iloc[:1000].reset_index(drop=True)

# Build structures
print("\nBuilding K-D Tree...")
kd_tree = KDTree()
kd_tree.build(points)

print("Building LSH...")
lsh = LSH(num_hashes=100, num_bands=5)  # Fewer bands = more lenient
lsh.build(texts)

# APPROACH 1: Lower the similarity threshold significantly
print("\n" + "="*70)
print("APPROACH 1: Very Low Similarity Threshold")
print("-"*70)

spatial_min = np.array([2019, 94, 4.0, 0])
spatial_max = np.array([2021, 100, 10.0, 50])

hybrid = HybridSearch(kd_tree, lsh)
results, stats = hybrid.hybrid_query(
    spatial_min, spatial_max,
    query_index=0,
    text_threshold=0.01,  # Very low threshold!
    max_results=5
)

print(f"\nResults with 0.01 threshold: {len(results)} matches")

# APPROACH 2: Use a query text that matches common terms
print("\n" + "="*70)
print("APPROACH 2: Query with Common Coffee Terms")
print("-"*70)

# Create a generic coffee description
generic_query = "chocolate sweet fruity bright acidity smooth body rich flavor coffee notes"

results2, stats2 = hybrid.hybrid_query(
    spatial_min, spatial_max,
    query_text=generic_query,
    text_threshold=0.05,
    max_results=5
)

print(f"\nResults with generic query: {len(results2)} matches")

# APPROACH 3: Show what's happening step by step
print("\n" + "="*70)
print("APPROACH 3: Debugging - What's Actually Happening")
print("-"*70)

# Get spatial results
spatial_results, _ = kd_tree.range_query(spatial_min, spatial_max)
spatial_indices = [idx for _, idx in spatial_results]
print(f"Spatial query found: {len(spatial_indices)} coffees")
print(f"First 5 indices: {spatial_indices[:5]}")

# Check if any of these have similar texts
print("\nChecking text similarity for spatial results...")
found_similar = 0
for idx in spatial_indices[:20]:  # Check first 20
    similar, _ = lsh.find_similar_by_index(idx, threshold=0.05, max_results=5)
    if similar:
        found_similar += 1
        print(f"Coffee {idx} has {len(similar)} similar texts")
        
print(f"\nTotal spatial results with similar texts: {found_similar}")

# APPROACH 4: Just return spatial results ranked by ANY text similarity
print("\n" + "="*70)
print("APPROACH 4: Spatial Results Ranked by Text Similarity")
print("-"*70)

# For each spatial result, compute similarity to query
query_idx = 0
query_sig = lsh.signatures[query_idx]

spatial_with_similarity = []
for point, idx in spatial_results[:50]:  # Check first 50
    if idx < len(lsh.signatures):
        similarity = lsh.minhash.jaccard_similarity(query_sig, lsh.signatures[idx])
        spatial_with_similarity.append((point, idx, similarity))

# Sort by similarity
spatial_with_similarity.sort(key=lambda x: x[2], reverse=True)

print(f"\nTop 5 spatial results by text similarity:")
for i, (point, idx, sim) in enumerate(spatial_with_similarity[:5]):
    coffee = df_subset.iloc[idx]
    print(f"{i+1}. {coffee['name']} (similarity: {sim:.1%})")
    print(f"   Year: {point[0]:.0f}, Rating: {point[1]:.0f}, Price: ${point[2]:.2f}")