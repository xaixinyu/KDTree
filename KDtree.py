from typing import List
from collections import namedtuple
import time


class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'

    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y


class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """
    '''
    Create the Node class for K-d Tree
    '''
    def _init_(self, location, left = None, right = None):
        self.location = location
        self.left = left
        self.right = right
        
        
    def __repr__(self):
        return f'{tuple(self)!r}'


class KDTree:
    """
    k-d tree
    """ 
    def __init__(self):
        self._root = None
        self._n = 0
        
    '''
    Create K-d Tree
    '''    
    def BuildKDtree(p, depth = 0):   
        try:
            k = len(p[0])
        except IndexError as e:
            return None
        axis = depth % k
        
        p.sort(key = lambda point: point[axis])
        median = len(points) // 2
        location = p[median]
        left = BuildKDtree(p[:median], depth + 1)
        right = BuildKDtree(p[median + 1:], depth + 1)
        return Node(location, left, right)
    
    
    def insert(self, p: List[Point]):
        """
        insert a list of points
        """
        def _insert(node, point, depth = 0):
            if node is None:
                return Node(point, None, None)
            axis = depth % 2
            if point[axis] < node.location[axis]:
                if node.left is None:
                    node.left = _insert(node.left, point, (depth + 1) % 2)
                else:
                    self._insert(node.left, point, depth + 1)
            else:
                if node.right is None: 
                    node.right = _insert(node.right, point, (depth + 1) % 2)
                else:
                    self._insert(node.right, point, depth + 1)
            return node
        self._root = _insert(self._root, p)
        self._n += len(p)
        pass   
    

    def range(self, rectangle: Rectangle) -> List[Point]:
        """
        range query
        """
        points = []
        def range_query(node, rectangle, depth = 0):
            if node is None:
                return points
            
            if rectangle.is_contains(node.location):
                points.append(node.location)
                
            axis = depth % 2
            if rectangle.lower[axis] <= node.location[axis]:
                points.extend(range_query(node.left, rectangle, depth + 1))
            if rectangle.upper[axis] <= node.location[axis]:
                points.extend(range_query(node.right, rectangle, depth + 1))
        range_query(self._root, rectangle)
        return points
        pass 
    
    def distance(self, point1, point2):
        dist = 0
        for i in range(len(point1)):
            dist += (point1[i] - point2[i]) ** 2
        dist = dist ** 0.5
        return dist

    
    def query(self, point):
        best = [None, float('inf')]
        def nearest_neighbor(node, point, depth = 0):
            if node is None:
                return 
            axis = depth & len(point)
            _dist = self.distance(point, node.point)
            if _dist < best[1]:
                best[0] = node.point
                best[1] = _dist
            if point[axis] < node.point[axis]:
                nearest_neighbor(node.left, point,depth+1)
            else:
                nearest_neighbor(node.right, point, depth+1)
            if abs(point[axis] - node.point[axis]) < best[1]:
                nearest_neighbor(node.right if point[axis] < node.point[axis]else node.left, point, depth+1)
        nearest_neighbor(self.root, 0)
        return best[0]  


def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])


def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # k-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)


if __name__ == '__main__':
    range_test()
    performance_test()

    
import matplotlib.pyplot as plt

points = [Point(x, y) for x in range(1000) for y in range(1000)]
x_axis = []
y_axis_kd= []
y_axis_naive = []
for n in range(1, 11):
    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    
    # naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points[:n*1000] if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    y_axis_naive.append(end - start)
    
    # k-d tree method
    kd = KDTree()
    kd.insert(points[:n*1000])
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    y_axis_kd.append(end - start)
    x_axis.append(n*1000)

plt.scatter(x_axis, y_axis_kd, label='K-D Tree')
plt.scatter(x_axis, y_axis_naive, label='Naive Method')
plt.xlabel('Number of Points')
plt.ylabel('Running Time (ms)')
plt.legend()
plt.show()

#the code on pdf
import matplotlib.pyplot as plt

num_points = [1000, 2000, 3000, 4000, 5000]
kd_tree_times = []
naive_times = []

for n in num_points:
    points = generate_random_points(n)
    start = time.time()
    kd_tree = KDTree()
    kd_tree.build(points)
    end = time.time()
    kd_tree_times.append(end - start)

    start = time.time()
    naive_search(points)
    end = time.time()
    naive_times.append(end - start)

plt.scatter(num_points, kd_tree_times, label='K-D Tree')
plt.scatter(num_points, naive_times, label='Naive')
plt.xlabel('Number of Points')
plt.ylabel('Running Time (s)')
plt.legend()
plt.show()
