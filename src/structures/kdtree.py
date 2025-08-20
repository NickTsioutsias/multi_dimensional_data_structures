"""
K-D Tree implementation for 4-dimensional data
"""
import numpy as np
import time

class KDNode:
    """A node in the K-D Tree"""
    def __init__(self, point, index, left=None, right=None):
        self.point = point  # The 4D point [year, rating, price, country_id]
        self.index = index  # Index in original dataset
        self.left = left    # Left child
        self.right = right   # Right child

class KDTree:
    """K-D Tree for 4-dimensional data"""
    
    def __init__(self):
        self.root = None
        self.k = 4  # 4 dimensions
        
    def build(self, points):
        """Build the tree from an array of points"""
        print(f"Building K-D Tree with {len(points)} points...")
        start_time = time.time()
        
        # Create indices array to track original positions
        indices = np.arange(len(points))
        
        # Build the tree
        self.root = self._build_recursive(points, indices, depth=0)
        
        build_time = time.time() - start_time
        print(f"K-D Tree built in {build_time:.3f} seconds")
        
    def _build_recursive(self, points, indices, depth):
        """Recursively build the tree"""
        if len(points) == 0:
            return None
            
        # Choose axis based on depth
        axis = depth % self.k
        
        # Sort points by the chosen axis
        sorted_order = np.argsort(points[:, axis])
        points = points[sorted_order]
        indices = indices[sorted_order]
        
        # Find median
        median_idx = len(points) // 2
        
        # Create node
        node = KDNode(
            point=points[median_idx],
            index=indices[median_idx]
        )
        
        # Build left and right subtrees
        node.left = self._build_recursive(
            points[:median_idx],
            indices[:median_idx],
            depth + 1
        )
        node.right = self._build_recursive(
            points[median_idx + 1:],
            indices[median_idx + 1:],
            depth + 1
        )
        
        return node
    
    def range_query(self, min_bounds, max_bounds):
        """
        Find all points within the given range
        min_bounds: [min_year, min_rating, min_price, min_country]
        max_bounds: [max_year, max_rating, max_price, max_country]
        """
        results = []
        self._range_query_recursive(self.root, min_bounds, max_bounds, 0, results)
        return results
    
    def _range_query_recursive(self, node, min_bounds, max_bounds, depth, results):
        """Recursively search for points in range"""
        if node is None:
            return
        
        # Check if current point is in range
        in_range = True
        for i in range(self.k):
            if node.point[i] < min_bounds[i] or node.point[i] > max_bounds[i]:
                in_range = False
                break
        
        if in_range:
            results.append((node.point, node.index))
        
        # Check which subtrees to explore
        axis = depth % self.k
        
        # Explore left subtree if needed
        if min_bounds[axis] <= node.point[axis]:
            self._range_query_recursive(node.left, min_bounds, max_bounds, depth + 1, results)
        
        # Explore right subtree if needed
        if max_bounds[axis] >= node.point[axis]:
            self._range_query_recursive(node.right, min_bounds, max_bounds, depth + 1, results)