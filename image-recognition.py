import numpy as np
import scipy.misc
import matplotlib.image as mpimg

class PNGMaker:
    def __init__(self, width = 1920, height = 1080):
        self.width = width
        self.height = height
        self.pixels = np.zeros((height, width, 3), dtype=np.uint8)

    # @PARAMS: array - 1D array of indices, each of which will be set to 1.
    def makeImageFromArray(self, array):
        self.pixels = array

        for index in array:
            self.pixels[index] = 1 # INDEXING BY OPPOSITE SIDES (width,height) <-- FIX LATER!!!