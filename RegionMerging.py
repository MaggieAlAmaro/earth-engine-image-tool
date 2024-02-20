
class ToVisitStack():
    def __init__(self):
        self.item = []

    def push(self, value):
        self.item.append(value)

    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []

class Visited():
    def __init__(self) -> None:
        self.visited : Pixel = [] 

    def hasVisited(self,v):
        return v in self.visited
        

class Pixel:
    def __init__(self,x,y ) -> None:
        self.x = x
        self.y = y


from PIL import Image
import numpy as np
class Merge:
    def __init__(self, x : np.array) -> None:
        self.arr = x
        self.size = self.arr.shape
        self.stack = ToVisitStack()


    def findMountain(self):
        value = 204
        mountainIndexes = np.where(self.arr == value)[0]
        # #TODO
        # for i in range(self.size[0]):
        #     for j in range(self.size[1]):
        #         if self.arr[i,j] == value:
        #             #expand mountain.

    def changeNeighbours(self, x, y):
        # for kernel of 3
        valueToMerge = 153
        valueToMerge2 = 102
        mountainValue = 204
        kernel = 3
        offset = kernel // 2
        startX = (offset - x) if (offset - x) >= 0 else 0 
        startY = (offset - y) if (offset - y) >= 0 else 0  #TODO edge pixels
        endX = (x + offset) if (x + offset) < self.size[0] else self.size[0] -1  #TODO edge pixels
        endY = (y + offset) if y + offset < self.size[1] else self.size[1] -1  #TODO edge pixels
        for i in range(startX,endX+1):
            for j in range(startY,endY+1):
                if(i == x and j == y):
                    continue
                if(self.arr[i,j] == valueToMerge2 or self.arr[i,j] == valueToMerge):
                    self.arr[i,j] = mountainValue
                    self.stack.push(Pixel(i,j))
                

    def propagate(self, startPixel : Pixel):
        self.stack.push(startPixel)
        while not self.stack.isEmpty():
            pixel = self.stack.pop()
            self.changeNeighbours(pixel.x,pixel.y)


    def propagateAllMountains(self):
        value = 204 
        mountainIndexes = np.where(self.arr == value)
        er = np.asarray(mountainIndexes).T
        print(er)
        print(er.shape)
        if(mountainIndexes == None):
            return 
        Pixel(er[0][0],er[0][1])

if __name__ == '__main__': 

    #arr = openSeg("output\segmentation-2024-02-12-16-10-54\img.png")
            
    # denoiseTests-2024-02-17-23-35-24
    denoisedImg = Image.open("output\denoiseTests-2024-02-19-16-06-35\img.png") #denoiseTests-2024-02-17-23-35-24\img.png") 
    arr = np.array(denoisedImg) 
    print(np.unique(arr))
    value = 204 

    # get indexes of where value is value
    mountainIndexes = np.where(arr == value)
    # get indexes in [[i,j]...[in,jn]] format
    er = np.asarray(mountainIndexes).T
    #access index
    # arr[i[0]][i[1]]

    # print(er)
    # print(er.shape)

