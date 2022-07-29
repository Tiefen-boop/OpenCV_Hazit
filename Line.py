import math


class Line:
    def __init__(self,start,end):
        self.startPoint=start
        self.endPoint=end

    def lineLength(self):
        return math.dist(self.endPoint,self.startPoint)