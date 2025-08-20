"""
Test Range Tree implementation
"""
from src.utils.data_loader import CoffeeDataLoader
from src.structures.rangetree import RangeTree
from src.structures.kdtree_complete import KDTree
from src.structures.quadtree import QuadTree
import numpy as np

print("="*60)
print("RANGE TREE TEST")
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
range_tree = RangeTree(dimensions=4)
range_tree.build(test_points)
print(f"Range Tree stats: {range_tree.get_stats()}")

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

results, query_time = range_tree.range_query(min_bounds, max_bounds)
print(f"\nRange Tree found {len(results)} points in {query_time:.4f} seconds")

# Show first 3 results
if results:
    print("\nFirst 3 results:")
    for i, (point, idx) in enumerate(results[:3]):
        coffee = df.iloc[idx]
        print(f"  {i+1}. {coffee['name'][:30]}... (Rating: {point[1]:.1f}, Price: ${point[2]:.2f})")

# 3. INSERT TEST
print("\n3. INSERT OPERATION TEST")
print("-"*40)

new_point = np.array([2020, 92, 7.5, 1])
print(f"Inserting point: {new_point}")

import time
start = time.time()
success = range_tree.insert(new_point, len(test_points))
insert_time = time.time() - start

print(f"Insert successful: {success}")
print(f"Insert time: {insert_time*1000:.3f}ms")
print(f"New tree size: {range_tree.size}")

# 4. COMPARISON WITH ALL STRUCTURES
print("\n4. COMPARISON OF ALL THREE STRUCTURES")
print("-"*40)

# Build all three structures
print("Building all structures...")
kd_tree = KDTree()
kd_tree.build(test_points)
kd_stats = kd_tree.get_stats()

quad_tree = QuadTree(capacity=10)
quad_tree.build(test_points)
quad_stats = quad_tree.get_stats()

# Run same query on all
kd_results, kd_time = kd_tree.range_query(min_bounds, max_bounds)
quad_results, quad_time = quad_tree.range_query(min_bounds, max_bounds)

# Summary table
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"{'Structure':<15} | {'Build (ms)':<10} | {'Query (ms)':<10} | {'Results':<8}")
print("-"*60)
print(f"{'K-D Tree':<15} | {kd_stats['build_time']*1000:<10.2f} | {kd_time*1000:<10.2f} | {len(kd_results):<8}")
print(f"{'Quad Tree':<15} | {'~8':<10} | {quad_time*1000:<10.2f} | {len(quad_results):<8}")
print(f"{'Range Tree':<15} | {range_tree.build_time*1000:<10.2f} | {query_time*1000:<10.2f} | {len(results):<8}")

# Find the fastest
times = [('K-D Tree', kd_time), ('Quad Tree', quad_time), ('Range Tree', query_time)]
fastest = min(times, key=lambda x: x[1])
print(f"\nFastest query: {fastest[0]} ({fastest[1]*1000:.2f}ms)")

print("\n" + "="*60)
print("RANGE TREE TESTING COMPLETE ✓")
print("="*60)