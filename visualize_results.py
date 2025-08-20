"""
Visualize the data and query results
"""
import matplotlib.pyplot as plt
from src.utils.data_loader import CoffeeDataLoader
from src.structures.kdtree import KDTree
import numpy as np

# Load data
loader = CoffeeDataLoader('data/simplified_coffee.csv')
df = loader.preprocess()
points = loader.get_points()

# Build tree
tree = KDTree()
tree.build(points)

# Run the same query
min_bounds = np.array([2019, 94, 4.0, 0])
max_bounds = np.array([2021, 100, 10.0, 50])
results = tree.range_query(min_bounds, max_bounds)

# Extract result points for visualization
result_indices = [idx for _, idx in results]
result_points = points[result_indices]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Coffee Data Distribution and Query Results', fontsize=16)

# Plot 1: Year vs Rating
axes[0, 0].scatter(points[:, 0], points[:, 1], alpha=0.3, label='All coffees')
if len(result_points) > 0:
    axes[0, 0].scatter(result_points[:, 0], result_points[:, 1], 
                      color='red', alpha=0.8, label='Query results')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Rating')
axes[0, 0].set_title('Year vs Rating')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Price vs Rating
axes[0, 1].scatter(points[:, 2], points[:, 1], alpha=0.3, label='All coffees')
if len(result_points) > 0:
    axes[0, 1].scatter(result_points[:, 2], result_points[:, 1], 
                      color='red', alpha=0.8, label='Query results')
axes[0, 1].set_xlabel('Price ($)')
axes[0, 1].set_ylabel('Rating')
axes[0, 1].set_title('Price vs Rating')
axes[0, 1].set_xlim(0, 50)  # Limit x-axis for better visibility
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Price distribution
axes[1, 0].hist(points[:, 2], bins=30, alpha=0.5, label='All coffees', color='blue')
if len(result_points) > 0:
    axes[1, 0].hist(result_points[:, 2], bins=15, alpha=0.7, label='Query results', color='red')
axes[1, 0].set_xlabel('Price ($)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Price Distribution')
axes[1, 0].set_xlim(0, 50)  # Limit for better visibility
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Rating distribution
axes[1, 1].hist(points[:, 1], bins=20, alpha=0.5, label='All coffees', color='blue')
if len(result_points) > 0:
    axes[1, 1].hist(result_points[:, 1], bins=10, alpha=0.7, label='Query results', color='red')
axes[1, 1].set_xlabel('Rating')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Rating Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kdtree_results.png', dpi=100)
plt.show()

print(f"\nVisualization saved as 'kdtree_results.png'")
print(f"Total coffees: {len(points)}")
print(f"Query results: {len(results)} coffees")
print(f"That's {len(results)/len(points)*100:.1f}% of all coffees")