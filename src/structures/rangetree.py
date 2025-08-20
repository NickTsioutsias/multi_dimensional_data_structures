"""
Range Tree implementation for 4D data
Range trees are optimized for orthogonal range queries
"""
import numpy as np
import time

class RangeNode:
    """A node in the Range Tree"""
    def __init__(self, points=None, indices=None, dim=0):
        self.points = points if points is not None else []
        self.indices = indices if indices is not None else []
        self.dim = dim
        self.value = None
        self.left = None
        self.right = None
        self.assoc_tree = None  # Associated structure for next dimension
        
class RangeTree:
    """Range Tree for efficient orthogonal range queries"""
    
    def __init__(self, dimensions=4):
        self.root = None
        self.dimensions = dimensions
        self.size = 0
        self.build_time = 0
        
    def build(self, points):
        """Build the Range Tree from points"""
        print(f"Building Range Tree with {len(points)} points...")
        start_time = time.time()
        
        indices = np.arange(len(points))
        self.size = len(points)
        
        # Build the main tree on first dimension
        self.root = self._build_tree(points, indices, 0)
        
        self.build_time = time.time() - start_time
        print(f"Range Tree built in {self.build_time:.3f} seconds")
    
    def _build_tree(self, points, indices, dim):
        """Recursively build a tree for the given dimension"""
        if len(points) == 0:
            return None
            
        if len(points) == 1:
            node = RangeNode(points, indices, dim)
            node.value = points[0][dim]
            
            # Build associated tree for next dimension if not last dimension
            if dim < self.dimensions - 1:
                node.assoc_tree = self._build_tree(points, indices, dim + 1)
            return node
        
        # Sort by current dimension
        sorted_order = np.argsort(points[:, dim])
        sorted_points = points[sorted_order]
        sorted_indices = indices[sorted_order]
        
        # Find median
        mid = len(sorted_points) // 2
        
        # Create node
        node = RangeNode(sorted_points, sorted_indices, dim)
        node.value = sorted_points[mid][dim]
        
        # Build left and right subtrees
        node.left = self._build_tree(
            sorted_points[:mid],
            sorted_indices[:mid],
            dim
        )
        node.right = self._build_tree(
            sorted_points[mid:],
            sorted_indices[mid:],
            dim
        )
        
        # Build associated tree for next dimension
        if dim < self.dimensions - 1:
            node.assoc_tree = self._build_tree(sorted_points, sorted_indices, dim + 1)
        
        return node
    
    def range_query(self, min_bounds, max_bounds):
        """Find all points within the given range"""
        start_time = time.time()
        results = []
        
        if self.root is not None:
            self._range_query_recursive(self.root, min_bounds, max_bounds, 0, results)
        
        query_time = time.time() - start_time
        return results, query_time
    
    def _range_query_recursive(self, node, min_bounds, max_bounds, dim, results):
        """Recursively search for points in range"""
        if node is None:
            return
        
        # If we're at the last dimension, check all points in this node
        if dim == self.dimensions - 1:
            for point, idx in zip(node.points, node.indices):
                if min_bounds[dim] <= point[dim] <= max_bounds[dim]:
                    # Check all dimensions
                    in_range = True
                    for d in range(self.dimensions):
                        if not (min_bounds[d] <= point[d] <= max_bounds[d]):
                            in_range = False
                            break
                    if in_range:
                        results.append((point, idx))
            return
        
        # If this is a leaf node
        if node.left is None and node.right is None:
            # Check if the point is in range
            if len(node.points) > 0:
                point = node.points[0]
                in_range = True
                for d in range(self.dimensions):
                    if not (min_bounds[d] <= point[d] <= max_bounds[d]):
                        in_range = False
                        break
                if in_range:
                    results.append((point, node.indices[0]))
            return
        
        # Check if we need to explore left subtree
        if node.left and min_bounds[dim] <= node.value:
            self._range_query_recursive(node.left, min_bounds, max_bounds, dim, results)
        
        # Check if we need to explore right subtree
        if node.right and max_bounds[dim] >= node.value:
            self._range_query_recursive(node.right, min_bounds, max_bounds, dim, results)
    
    def insert(self, point, index):
        """Insert a new point (simplified version)"""
        if self.root is None:
            self.root = RangeNode(np.array([point]), np.array([index]), 0)
            self.root.value = point[0]
            self.size = 1
            return True
        
        # Simple insertion at leaf level
        self._insert_recursive(self.root, point, index, 0)
        self.size += 1
        return True
    
    def _insert_recursive(self, node, point, index, dim):
        """Recursively insert a point"""
        if node.left is None and node.right is None:
            # Leaf node - add point to this node's collection
            node.points = np.vstack([node.points, point])
            node.indices = np.append(node.indices, index)
            return
        
        # Navigate to appropriate subtree
        if point[dim] < node.value:
            if node.left:
                self._insert_recursive(node.left, point, index, dim)
            else:
                node.left = RangeNode(np.array([point]), np.array([index]), dim)
                node.left.value = point[dim]
        else:
            if node.right:
                self._insert_recursive(node.right, point, index, dim)
            else:
                node.right = RangeNode(np.array([point]), np.array([index]), dim)
                node.right.value = point[dim]
    
    def get_stats(self):
        """Get tree statistics"""
        def count_nodes(node):
            if node is None:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        def get_height(node):
            if node is None:
                return 0
            return 1 + max(get_height(node.left), get_height(node.right))
        
        return {
            'size': self.size,
            'dimensions': self.dimensions,
            'nodes': count_nodes(self.root),
            'height': get_height(self.root),
            'build_time': self.build_time
        }