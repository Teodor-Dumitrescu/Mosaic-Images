
"""
    PROIECT MOZAIC
"""

# Parametrii algoritmului sunt definiti in clasa Parameters.
from parameters import *
from build_mosaic import *

# numele imaginii care va fi transformata in mozaic
image_path = './../data/imaginiTest/frog.png'
# daca imaginea este grayscale
grayscale = False
cifar = True

params = Parameters(image_path, grayscale, cifar)

# cifar-10 class: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
params.cifar_type = b'frog'
# directorul cu imagini folosite pentru realizarea mozaicului
params.small_images_dir = './../data/colectie/'
# tipul imaginilor din director
params.image_type = 'png'
# numarul de piese ale mozaicului pe orizontala
# pe verticala vor fi calcultate dinamic a.i sa se pastreze raportul
params.num_pieces_horizontal = 75
# afiseaza piesele de mozaic dupa citirea lor
params.show_small_images = True
# modul de aranjarea a pieselor mozaicului
# optiuni: 'aleator', 'caroiaj'
params.layout = 'caroiaj'
# criteriul dupa care se realizeaza mozaicul
# optiuni: 'aleator', 'distantaCuloareMedie'
params.criterion = 'distantaCuloareMedie'
# daca params.layout == 'caroiaj', sa se foloseasca piese hexagonale
params.hexagon = False

img_mosaic = build_mosaic(params)
cv.imwrite('mozaic.png', img_mosaic)
cv.imwrite('resized.png', params.image_resized)