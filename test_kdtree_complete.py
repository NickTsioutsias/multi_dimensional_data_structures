"""
Test complete K-D Tree operations: Build, Insert, Delete, Query
"""
from src.utils.data_loader import CoffeeDataLoader
from src.structures.kdtree_complete import KDTree
import numpy as np
import time

print("="*60)
print("COMPLETE K-D TREE TEST")
print("="*60)

# Load data
loader = CoffeeDataLoader('data/simplified_coffee.csv')
df = loader.preprocess()
points = loader.get_points()

# Use first 1000 points for initial tree
initial_points = points[:1000]
test_points = points[1000:1010]  # Points for insert/delete testing

# 1. BUILD TEST
print("\n1. BUILD OPERATION")
print("-"*40)
tree = KDTree()
tree.build(initial_points)
print(f"Tree stats: {tree.get_stats()}")

# 2. RANGE QUERY TEST
print("\n2. RANGE QUERY TEST")
print("-"*40)
min_bounds = np.array([2019, 90, 5.0, 0])
max_bounds = np.array([2021, 95, 15.0, 50])
results, query_time = tree.range_query(min_bounds, max_bounds)
print(f"Query found {len(results)} points in {query_time:.4f} seconds")

# 3. INSERT TEST
print("\n3. INSERT OPERATION TEST")
print("-"*40)
print(f"Tree size before insert: {tree.size}")

# Insert 5 new points
insert_times = []
for i in range(5):
    point = test_points[i]
    start = time.time()
    success = tree.insert(point, 1000 + i)
    insert_time = time.time() - start
    insert_times.append(insert_time)
    print(f"  Inserted point {i+1}: {success} (time: {insert_time:.5f}s)")

print(f"Tree size after inserts: {tree.size}")
print(f"Average insert time: {np.mean(insert_times):.5f} seconds")

# 4. DELETE TEST
print("\n4. DELETE OPERATION TEST")
print("-"*40)
print(f"Tree size before delete: {tree.size}")

# Delete 3 points
delete_times = []
for i in range(3):
    point = test_points[i]
    start = time.time()
    success = tree.delete(point)
    delete_time = time.time() - start
    delete_times.append(delete_time)
    print(f"  Deleted point {i+1}: {success} (time: {delete_time:.5f}s)")

print(f"Tree size after deletes: {tree.size}")
print(f"Average delete time: {np.mean(delete_times):.5f} seconds")

# 5. FINAL STATS
print("\n5. FINAL TREE STATISTICS")
print("-"*40)
final_stats = tree.get_stats()
for key, value in final_stats.items():
    print(f"  {key}: {value}")

# 6. PERFORMANCE COMPARISON
print("\n6. PERFORMANCE SUMMARY")
print("-"*40)
print(f"Build time (1000 points): ~{tree.size/1000:.3f}ms per point")
print(f"Query time: {query_time*1000:.2f}ms")
print(f"Insert time: {np.mean(insert_times)*1000:.2f}ms average")
print(f"Delete time: {np.mean(delete_times)*1000:.2f}ms average")

print("\n" + "="*60)
print("K-D TREE TESTING COMPLETE ✓")
print("="*60)