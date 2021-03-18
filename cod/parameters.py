import cv2 as cv
import numpy as np


# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:
    def __init__(self, image_path, grayscale, cifar):
        self.image_path = image_path
        self.grayscale = grayscale
        if self.grayscale:
            # read the image
            self.image = cv.imread(image_path)
            # convert it to grayscale
            gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            # make it have 3 channels so the code for color images works in this case too
            self.image = np.zeros_like(self.image)
            self.image[:, :, 0] = gray.copy()
            self.image[:, :, 1] = gray.copy()
            self.image[:, :, 2] = gray.copy()
        else:
            self.image = cv.imread(image_path)
        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)

        self.cifar = cifar
        self.cifar_type = b'bird'
        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None

