class Station:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def getCoordonate(self):
        return self.x, self.y
