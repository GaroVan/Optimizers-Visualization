class vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar:float):
        return vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return vector(self.y * other.z - self.z * other.y,
                        self.z * other.x - self.x * other.z,
                        self.x * other.y - self.y * other.x)
    
    def __truediv__(self, scalar:float):
        return vector(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    def __repr__(self):
        return f"vector({self.x}, {self.y}, {self.z})"


