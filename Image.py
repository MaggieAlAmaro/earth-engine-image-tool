
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import typing

class MyImage(ABC):
    def __init__(self, image_filename) -> None:
        self.image_filename = image_filename
        self.image = Image.open(image_filename)
        self.data = np.array(self.image)
        self.size = self.data.shape #self.image.size -> is backwards

    @abstractmethod
    def post_process_step(self, processedData=None) -> Image: 
        pass
        

class RGBImage(MyImage):
    def __init__(self, image_filename) -> None:
        super().__init__(image_filename)

    def post_process_step(self, processedData=None) -> Image: 
        pass




class GrayscaleImage(MyImage):
    def __init__(self, image_filename) -> None:
        super().__init__(image_filename)
        
    def post_process_step(self, processedData=None) -> Image: 
        return Image.fromarray(processedData)



class RGBAImage(MyImage):
    def __init__(self, image_filename, proccessA=False) -> None:
        super().__init__(image_filename)
        #rgba.separateAandRGB(image_filename)
        self.r, self.g, self.b, self.a = self.image.split()
        self.processA = proccessA
        
        if proccessA:
            self.data = np.array(self.a)
        else:
            rgb = Image.merge('RGB', (self.r, self.g, self.b))
            self.data = np.array(rgb)

    def post_process_step(self, processedData=None) -> Image: 
        #processedData = self.data if processedData.any() == None else processedData
        processedData = Image.fromarray(processedData)
        if self.processA:
            processedData = Image.merge('RGBA', (self.r, self.g, self.b, processedData))
        else:
            processedData = Image.merge('RGBA', (processedData, self.a))
        return processedData





