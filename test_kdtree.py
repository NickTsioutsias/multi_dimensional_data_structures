"""
Test K-D Tree with coffee data
"""
from src.utils.data_loader import CoffeeDataLoader
from src.structures.kdtree import KDTree
import numpy as np
import time

# Load data
print("Loading data...")
loader = CoffeeDataLoader('data/simplified_coffee.csv')
df = loader.preprocess()
points = loader.get_points()

print(f"\nLoaded {len(points)} coffee reviews")
print("="*50)

# Build K-D Tree
tree = KDTree()
tree.build(points)

print("="*50)
print("\nTesting Range Query")
print("-"*50)

# Example query: Find coffees from 2019-2021 with rating > 94, price $4-$10
min_bounds = np.array([2019, 94, 4.0, 0])    # min values
max_bounds = np.array([2021, 100, 10.0, 50]) # max values (using 50 for country to include all)

print(f"Query parameters:")
print(f"  Years: {min_bounds[0]:.0f} - {max_bounds[0]:.0f}")
print(f"  Rating: {min_bounds[1]:.1f} - {max_bounds[1]:.1f}")
print(f"  Price: ${min_bounds[2]:.2f} - ${max_bounds[2]:.2f}")

# Run query
start_time = time.time()
results = tree.range_query(min_bounds, max_bounds)
query_time = time.time() - start_time

print(f"\nFound {len(results)} coffees in {query_time:.4f} seconds")

# Show first 5 results
if results:
    print(f"\nShowing first {min(5, len(results))} results:")
    print("-"*50)
    
    for i, (point, idx) in enumerate(results[:5]):
        coffee = df.iloc[idx]
        print(f"\n{i+1}. {coffee['name']}")
        print(f"   Roaster: {coffee['roaster']}")
        print(f"   Year: {point[0]:.0f}, Rating: {point[1]:.1f}, Price: ${point[2]:.2f}")
        print(f"   Country: {coffee['loc_country']}")

print("\n" + "="*50)
print("K-D Tree test complete!")