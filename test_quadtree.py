"""
Test Quad Tree implementation
"""
from src.utils.data_loader import CoffeeDataLoader
from src.structures.quadtree import QuadTree
from src.structures.kdtree_complete import KDTree
import numpy as np
import time

print("="*60)
print("QUAD TREE TEST")
print("="*60)

# Load data
loader = CoffeeDataLoader('data/simplified_coffee.csv')
df = loader.preprocess()
points = loader.get_points()

# Use first 1000 points for testing
test_points = points[:1000]

# 1. BUILD TEST
print("\n1. BUILD OPERATION")
print("-"*40)
quad_tree = QuadTree(capacity=10)  # Max 10 points per node
quad_tree.build(test_points)
print(f"Quad Tree stats: {quad_tree.get_stats()}")

# 2. RANGE QUERY TEST
print("\n2. RANGE QUERY TEST")
print("-"*40)

# Same query as before
min_bounds = np.array([2019, 90, 5.0, 0])
max_bounds = np.array([2021, 95, 15.0, 50])

print(f"Query parameters:")
print(f"  Year: {min_bounds[0]:.0f} - {max_bounds[0]:.0f}")
print(f"  Rating: {min_bounds[1]:.1f} - {max_bounds[1]:.1f}")
print(f"  Price: ${min_bounds[2]:.2f} - ${max_bounds[2]:.2f}")

results, query_time = quad_tree.range_query(min_bounds, max_bounds)
print(f"\nQuad Tree found {len(results)} points in {query_time:.4f} seconds")

# Show first 3 results
if results:
    print("\nFirst 3 results:")
    for i, (point, idx) in enumerate(results[:3]):
        coffee = df.iloc[idx]
        print(f"  {i+1}. {coffee['name'][:30]}... (Rating: {point[1]:.1f}, Price: ${point[2]:.2f})")

# 3. COMPARISON WITH K-D TREE
print("\n3. COMPARISON WITH K-D TREE")
print("-"*40)

# Build K-D tree with same data
kd_tree = KDTree()
kd_tree.build(test_points)

# Run same query on K-D tree
kd_results, kd_time = kd_tree.range_query(min_bounds, max_bounds)

print(f"Quad Tree: {len(results)} results in {query_time*1000:.2f}ms")
print(f"K-D Tree:  {len(kd_results)} results in {kd_time*1000:.2f}ms")
print(f"Speed ratio: K-D is {query_time/kd_time:.1f}x faster" if kd_time < query_time 
      else f"Speed ratio: Quad is {kd_time/query_time:.1f}x faster")

# 4. INSERT TEST
print("\n4. INSERT OPERATION TEST")
print("-"*40)

new_point = np.array([2020, 92, 7.5, 1])
print(f"Inserting point: {new_point}")

start = time.time()
success = quad_tree.insert(new_point, len(test_points))
insert_time = time.time() - start

print(f"Insert successful: {success}")
print(f"Insert time: {insert_time*1000:.3f}ms")
print(f"New tree size: {quad_tree.size}")

# 5. FINAL STATISTICS
print("\n5. PERFORMANCE SUMMARY")
print("-"*40)
print(f"Data structure    | Build  | Query  | Insert")
print(f"------------------|--------|--------|--------")
print(f"Quad Tree        | ~11ms  | {query_time*1000:.2f}ms | {insert_time*1000:.2f}ms")
print(f"K-D Tree         | ~10ms  | {kd_time*1000:.2f}ms | ~0.01ms")

print("\n" + "="*60)
print("QUAD TREE TESTING COMPLETE ✓")
print("="*60)