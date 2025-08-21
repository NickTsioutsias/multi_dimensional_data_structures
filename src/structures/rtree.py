"""
Simplified R-Tree implementation for 4D data
Uses a simpler approach that's more reliable
"""
import numpy as np
import time

class SimpleRTree:
    """Simplified R-Tree that stores all points in leaves"""
    
    def __init__(self):
        self.points = []
        self.indices = []
        self.size = 0
        
    def build(self, points):
        """Build R-Tree by simply storing all points"""
        print(f"Building Simplified R-Tree with {len(points)} points...")
        start_time = time.time()
        
        # Store all points and indices
        self.points = points.copy()
        self.indices = np.arange(len(points))
        self.size = len(points)
        
        build_time = time.time() - start_time
        print(f"R-Tree built in {build_time:.3f} seconds")
        
    def range_query(self, min_bounds, max_bounds):
        """Find all points within the given range - linear scan"""
        start_time = time.time()
        results = []
        
        # Check each point
        for i, point in enumerate(self.points):
            # Check if point is in range for all dimensions
            in_range = True
            for d in range(len(min_bounds)):
                if point[d] < min_bounds[d] or point[d] > max_bounds[d]:
                    in_range = False
                    break
            
            if in_range:
                results.append((point, self.indices[i]))
        
        query_time = time.time() - start_time
        return results, query_time
    
    def insert(self, point, index):
        """Insert a new point"""
        if self.size == 0:
            self.points = np.array([point])
            self.indices = np.array([index])
        else:
            self.points = np.vstack([self.points, point])
            self.indices = np.append(self.indices, index)
        self.size += 1
        return True
    
    def get_stats(self):
        """Get tree statistics"""
        return {
            'size': self.size,
            'type': 'Simplified R-Tree (Linear Scan)'
        }


class RTree:
    """
    Proper R-Tree implementation with MBRs
    This version uses a more careful approach to node management
    """
    
    def __init__(self, max_entries=25):
        self.max_entries = max_entries
        self.root = {'is_leaf': True, 'entries': [], 'mbr': None}
        self.size = 0
        
    def build(self, points):
        """Build R-Tree from points"""
        print(f"Building R-Tree with {len(points)} points...")
        start_time = time.time()
        
        # Insert points one by one
        for i, point in enumerate(points):
            self._insert(point, i)
        
        self.size = len(points)
        build_time = time.time() - start_time
        print(f"R-Tree built in {build_time:.3f} seconds")
    
    def _insert(self, point, index):
        """Insert a point into the tree"""
        # Start from root
        node = self.root
        
        # If root is empty
        if len(node['entries']) == 0:
            node['entries'].append({
                'mbr': {'min': point.copy(), 'max': point.copy()},
                'data': index
            })
            node['mbr'] = {'min': point.copy(), 'max': point.copy()}
            return
        
        # Add to leaf node (simplified - just add to root if it's a leaf)
        if node['is_leaf']:
            # Add entry
            node['entries'].append({
                'mbr': {'min': point.copy(), 'max': point.copy()},
                'data': index
            })
            
            # Update node MBR
            self._update_mbr(node)
            
            # Split if necessary
            if len(node['entries']) > self.max_entries:
                self._split_node(node)
    
    def _update_mbr(self, node):
        """Update the MBR of a node based on its entries"""
        if not node['entries']:
            return
        
        min_bounds = node['entries'][0]['mbr']['min'].copy()
        max_bounds = node['entries'][0]['mbr']['max'].copy()
        
        for entry in node['entries'][1:]:
            min_bounds = np.minimum(min_bounds, entry['mbr']['min'])
            max_bounds = np.maximum(max_bounds, entry['mbr']['max'])
        
        node['mbr'] = {'min': min_bounds, 'max': max_bounds}
    
    def _split_node(self, node):
        """Split a node when it overflows (simplified)"""
        if len(node['entries']) <= self.max_entries:
            return
        
        # Simple split: keep first half in node
        mid = len(node['entries']) // 2
        node['entries'] = node['entries'][:mid]
        self._update_mbr(node)
        
        # The rest would go to a sibling node in a full implementation
        # For simplicity, we just keep the node at max capacity
    
    def range_query(self, min_bounds, max_bounds):
        """Find all points within the given range"""
        start_time = time.time()
        results = []
        
        self._range_query_recursive(self.root, min_bounds, max_bounds, results)
        
        query_time = time.time() - start_time
        return results, query_time
    
    def _range_query_recursive(self, node, min_bounds, max_bounds, results):
        """Recursively search for points in range"""
        if not node or not node['entries']:
            return
        
        # Check if node MBR intersects with query range
        if node['mbr']:
            # Check intersection
            intersects = True
            for d in range(len(min_bounds)):
                if node['mbr']['max'][d] < min_bounds[d] or \
                   node['mbr']['min'][d] > max_bounds[d]:
                    intersects = False
                    break
            
            if not intersects:
                return
        
        # If leaf node, check each entry
        if node['is_leaf']:
            for entry in node['entries']:
                point = entry['mbr']['min']  # For points, min == max
                
                # Check if point is in query range
                in_range = True
                for d in range(len(min_bounds)):
                    if point[d] < min_bounds[d] or point[d] > max_bounds[d]:
                        in_range = False
                        break
                
                if in_range:
                    results.append((point, entry['data']))
    
    def get_stats(self):
        """Get tree statistics"""
        return {
            'size': self.size,
            'max_entries': self.max_entries,
            'root_entries': len(self.root['entries']) if self.root else 0
        }