"""
Complete K-D Tree implementation with insert and delete operations
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
    """Complete K-D Tree for 4-dimensional data"""
    
    def __init__(self):
        self.root = None
        self.k = 4  # 4 dimensions
        self.size = 0
        
    def build(self, points):
        """Build the tree from an array of points"""
        print(f"Building K-D Tree with {len(points)} points...")
        start_time = time.time()
        
        # Create indices array to track original positions
        indices = np.arange(len(points))
        
        # Build the tree
        self.root = self._build_recursive(points, indices, depth=0)
        self.size = len(points)
        
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
    
    def insert(self, point, index):
        """Insert a new point into the tree"""
        if self.root is None:
            self.root = KDNode(point, index)
            self.size = 1
            return True
        
        # Find where to insert
        success = self._insert_recursive(self.root, point, index, 0)
        if success:
            self.size += 1
        return success
    
    def _insert_recursive(self, node, point, index, depth):
        """Recursively insert a point"""
        if node is None:
            return KDNode(point, index)
        
        # Choose axis
        axis = depth % self.k
        
        # Compare and go left or right
        if point[axis] < node.point[axis]:
            child = self._insert_recursive(node.left, point, index, depth + 1)
            if node.left is None:
                node.left = child
                return True
            return child
        else:
            child = self._insert_recursive(node.right, point, index, depth + 1)
            if node.right is None:
                node.right = child
                return True
            return child
    
    def delete(self, point):
        """Delete a point from the tree"""
        self.root, deleted = self._delete_recursive(self.root, point, 0)
        if deleted:
            self.size -= 1
        return deleted
    
    def _delete_recursive(self, node, point, depth):
        """Recursively delete a point"""
        if node is None:
            return None, False
        
        # Check if this is the node to delete
        if np.array_equal(node.point, point):
            # Case 1: Node has right subtree
            if node.right is not None:
                # Find minimum in right subtree
                min_node = self._find_min(node.right, depth % self.k, depth + 1)
                node.point = min_node.point
                node.index = min_node.index
                node.right, _ = self._delete_recursive(node.right, min_node.point, depth + 1)
            # Case 2: Node has only left subtree
            elif node.left is not None:
                # Find minimum in left subtree
                min_node = self._find_min(node.left, depth % self.k, depth + 1)
                node.point = min_node.point
                node.index = min_node.index
                node.left, _ = self._delete_recursive(node.left, min_node.point, depth + 1)
            # Case 3: Node is a leaf
            else:
                return None, True
            
            return node, True
        
        # Not the node to delete, continue searching
        axis = depth % self.k
        if point[axis] < node.point[axis]:
            node.left, deleted = self._delete_recursive(node.left, point, depth + 1)
        else:
            node.right, deleted = self._delete_recursive(node.right, point, depth + 1)
        
        return node, deleted
    
    def _find_min(self, node, dim, depth):
        """Find minimum node along a dimension"""
        if node is None:
            return None
        
        axis = depth % self.k
        
        if dim == axis:
            # Only need to check left subtree
            if node.left is None:
                return node
            return self._find_min(node.left, dim, depth + 1)
        
        # Need to check both subtrees
        left_min = self._find_min(node.left, dim, depth + 1)
        right_min = self._find_min(node.right, dim, depth + 1)
        
        # Find the minimum among node, left_min, and right_min
        min_node = node
        if left_min and left_min.point[dim] < min_node.point[dim]:
            min_node = left_min
        if right_min and right_min.point[dim] < min_node.point[dim]:
            min_node = right_min
        
        return min_node
    
    def range_query(self, min_bounds, max_bounds):
        """Find all points within the given range"""
        results = []
        start_time = time.time()
        self._range_query_recursive(self.root, min_bounds, max_bounds, 0, results)
        query_time = time.time() - start_time
        return results, query_time
    
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
    
    def get_stats(self):
        """Get tree statistics"""
        height = self._get_height(self.root)
        return {
            'size': self.size,
            'height': height,
            'dimensions': self.k,
            'balanced': self._is_balanced(self.root)
        }
    
    def _get_height(self, node):
        """Calculate tree height"""
        if node is None:
            return 0
        return 1 + max(self._get_height(node.left), self._get_height(node.right))
    
    def _is_balanced(self, node):
        """Check if tree is balanced"""
        if node is None:
            return True
        
        left_height = self._get_height(node.left)
        right_height = self._get_height(node.right)
        
        if abs(left_height - right_height) > 1:
            return False
        
        return self._is_balanced(node.left) and self._is_balanced(node.right)