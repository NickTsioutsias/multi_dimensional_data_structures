"""
Quad Tree implementation for 4D data (Fixed version)
Note: Quad trees work best with 2D data, so we'll project our 4D data to 2D pairs
"""
import numpy as np
import time

class QuadNode:
    """A node in the Quad Tree"""
    def __init__(self, x_min, x_max, y_min, y_max, capacity=10, max_depth=20, depth=0):
        # Boundaries
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        # Node properties
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.points = []  # List of (point, index) tuples
        self.divided = False
        
        # Children (quadrants)
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None
    
    def insert(self, point, index):
        """Insert a point into this node"""
        # Check if point is within boundaries
        if not (self.x_min <= point[0] <= self.x_max and 
                self.y_min <= point[1] <= self.y_max):
            return False
        
        # If we haven't reached capacity OR we're at max depth, just add the point
        if len(self.points) < self.capacity or self.depth >= self.max_depth:
            self.points.append((point, index))
            return True
        
        # Otherwise, subdivide if we haven't already
        if not self.divided:
            self.subdivide()
        
        # Try to insert into the appropriate child
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        
        if point[0] >= x_mid:
            if point[1] >= y_mid:
                return self.northeast.insert(point, index)
            else:
                return self.southeast.insert(point, index)
        else:
            if point[1] >= y_mid:
                return self.northwest.insert(point, index)
            else:
                return self.southwest.insert(point, index)
    
    def subdivide(self):
        """Subdivide this node into 4 quadrants"""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        
        # Create four children with increased depth
        self.northeast = QuadNode(x_mid, self.x_max, y_mid, self.y_max, 
                                 self.capacity, self.max_depth, self.depth + 1)
        self.northwest = QuadNode(self.x_min, x_mid, y_mid, self.y_max, 
                                 self.capacity, self.max_depth, self.depth + 1)
        self.southeast = QuadNode(x_mid, self.x_max, self.y_min, y_mid, 
                                 self.capacity, self.max_depth, self.depth + 1)
        self.southwest = QuadNode(self.x_min, x_mid, self.y_min, y_mid, 
                                 self.capacity, self.max_depth, self.depth + 1)
        
        self.divided = True
        
        # Move existing points to children
        old_points = self.points.copy()
        self.points = []
        
        for point, idx in old_points:
            x_mid = (self.x_min + self.x_max) / 2
            y_mid = (self.y_min + self.y_max) / 2
            
            if point[0] >= x_mid:
                if point[1] >= y_mid:
                    self.northeast.insert(point, idx)
                else:
                    self.southeast.insert(point, idx)
            else:
                if point[1] >= y_mid:
                    self.northwest.insert(point, idx)
                else:
                    self.southwest.insert(point, idx)
    
    def query_range(self, x_min, x_max, y_min, y_max, results):
        """Find all points within the given range"""
        # Check if this node intersects with query range
        if not (x_min <= self.x_max and x_max >= self.x_min and
                y_min <= self.y_max and y_max >= self.y_min):
            return
        
        # Check points in this node
        for point, idx in self.points:
            if (x_min <= point[0] <= x_max and 
                y_min <= point[1] <= y_max):
                results.append((point, idx))
        
        # If subdivided, query children
        if self.divided:
            self.northeast.query_range(x_min, x_max, y_min, y_max, results)
            self.northwest.query_range(x_min, x_max, y_min, y_max, results)
            self.southeast.query_range(x_min, x_max, y_min, y_max, results)
            self.southwest.query_range(x_min, x_max, y_min, y_max, results)


class QuadTree:
    """Quad Tree for multi-dimensional data (projects to 2D)"""
    
    def __init__(self, capacity=10, max_depth=20):
        self.root = None
        self.capacity = capacity
        self.max_depth = max_depth
        self.size = 0
        self.original_points = None  # Store original 4D points
        
    def build(self, points):
        """
        Build the Quad Tree from 4D points
        We'll use first two dimensions for spatial indexing
        """
        print(f"Building Quad Tree with {len(points)} points...")
        start_time = time.time()
        
        self.original_points = points.copy()
        
        # Find boundaries for first two dimensions
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        # Add small margin to avoid boundary issues
        margin = 0.01
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # Create root node
        self.root = QuadNode(x_min, x_max, y_min, y_max, self.capacity, self.max_depth)
        
        # Insert all points (using first 2 dimensions)
        for i, point in enumerate(points):
            success = self.root.insert(point[:2], i)
            if not success:
                print(f"Warning: Failed to insert point {i}")
        
        self.size = len(points)
        
        build_time = time.time() - start_time
        print(f"Quad Tree built in {build_time:.3f} seconds")
    
    def range_query(self, min_bounds, max_bounds):
        """
        Find all points within the given 4D range
        First filter by dimensions 0-1, then check dimensions 2-3
        """
        if self.root is None:
            return [], 0
        
        start_time = time.time()
        
        # Query using first two dimensions
        results_2d = []
        self.root.query_range(
            min_bounds[0], max_bounds[0],  # x range (dimension 0)
            min_bounds[1], max_bounds[1],  # y range (dimension 1)
            results_2d
        )
        
        # Filter results by checking dimensions 2 and 3
        final_results = []
        for point_2d, idx in results_2d:
            original_point = self.original_points[idx]
            if (min_bounds[2] <= original_point[2] <= max_bounds[2] and
                min_bounds[3] <= original_point[3] <= max_bounds[3]):
                final_results.append((original_point, idx))
        
        query_time = time.time() - start_time
        return final_results, query_time
    
    def insert(self, point, index):
        """Insert a new 4D point"""
        if self.root is None:
            return False
        
        # Add to original points
        if self.original_points is None:
            self.original_points = np.array([point])
        else:
            self.original_points = np.vstack([self.original_points, point])
        
        # Insert using first 2 dimensions
        success = self.root.insert(point[:2], index)
        if success:
            self.size += 1
        return success
    
    def get_stats(self):
        """Get tree statistics"""
        def count_nodes(node):
            if node is None:
                return 0
            count = 1
            if node.divided:
                count += count_nodes(node.northeast)
                count += count_nodes(node.northwest)
                count += count_nodes(node.southeast)
                count += count_nodes(node.southwest)
            return count
        
        def count_points(node):
            if node is None:
                return 0
            count = len(node.points)
            if node.divided:
                count += count_points(node.northeast)
                count += count_points(node.northwest)
                count += count_points(node.southeast)
                count += count_points(node.southwest)
            return count
        
        def get_depth(node):
            if node is None:
                return 0
            if not node.divided:
                return 1
            return 1 + max(
                get_depth(node.northeast),
                get_depth(node.northwest),
                get_depth(node.southeast),
                get_depth(node.southwest)
            )
        
        return {
            'size': self.size,
            'nodes': count_nodes(self.root),
            'depth': get_depth(self.root),
            'total_points': count_points(self.root),
            'capacity_per_node': self.capacity,
            'max_depth': self.max_depth
        }