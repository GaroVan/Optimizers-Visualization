import numpy as np

def order_vertices_clockwise(vertices):
    """
    Ensure the vertices are in clockwise order for bilinear interpolation.
    Assumes the quadrilateral is roughly rectangular.
    """
    assert len(vertices) == 4, "Must provide 4 vertices"
    assert all(len(v) == 3 for v in vertices), "Vertices must be 3D"
    
    # Sort by x first (left to right)
    vertices.sort(key=lambda v: v[0])
    
    # Sort the two left-most and right-most vertices by y to get bottom-left, bottom-right, etc.
    left = sorted(vertices[:2], key=lambda v: v[1])  # Bottom-left and top-left
    right = sorted(vertices[2:], key=lambda v: v[1]) # Bottom-right and top-right
    
    # Return ordered vertices (P1, P2, P3, P4 in clockwise order)
    return left[0], right[0], right[1], left[1]

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def grid_to_coordinate(r, min, max):
    """
    Convert a grid index to a coordinate in a given range.
    """
    assert min < max, "min must be less than max"
    assert r >= 0 and r <= 1, "r must be between 0 and 1"
    return min + r * (max - min)