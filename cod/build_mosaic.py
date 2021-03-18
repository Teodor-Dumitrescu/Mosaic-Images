import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
from add_pieces_mosaic import *
from parameters import *


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i

    # if we are using the flower collection
    if not params.cifar:
        images = []
        images_names = os.listdir(params.small_images_dir)
        for image_name in images_names:
            if params.grayscale:
                # read small image
                image = cv.imread(params.small_images_dir + image_name)
                # convert it to grayscale
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                # make it have 3 channels so the code for color images works in this case too
                image = np.zeros_like(image)
                image[:, :, 0] = gray.copy()
                image[:, :, 1] = gray.copy()
                image[:, :, 2] = gray.copy()
            else:
                image = cv.imread(params.small_images_dir + image_name)
            images.append(image)

        # citeste imaginile din director

        if params.show_small_images:
            for i in range(4):
                for j in range(4):
                    plt.subplot(4, 4, i * 4 + j + 1)
                    # OpenCV reads images in BGR format, matplotlib reads images in RBG format
                    im = images[i * 4 + j].copy()
                    # BGR to RGB, swap the channels
                    im = im[:, :, [2, 1, 0]]
                    plt.imshow(im)
            plt.show()

        params.small_images = np.array(images)
    # if we are using the Cifar-10 collection
    else:
        cifar_path = './../data/cifar-10-batches-py/data/'
        # get the list of cifar files with data
        cifar_files_names = os.listdir(cifar_path)
        # create a list with data from each cifar batch
        cifar_data = np.zeros((60000, 32, 32, 3))
        # create a list with labels for each image
        cifar_labels = []
        for i, cifar_file in enumerate(cifar_files_names):
            cifar_dict = unpickle(cifar_path + cifar_file)
            # the data comes in a shape like (nr_images, height * width * nr_channels), also the values for channels
            # are not alternating, they are like: (R,R,...,R,G,G,...G,B,B,...,B)
            # we have to reshape it so our algorithm can manipulate it
            data = np.zeros(shape=(cifar_dict[b'data'].shape[0], 32, 32, 3))
            data[:, :, :, 0] = cifar_dict[b'data'][:, 0: int(cifar_dict[b'data'].shape[1] / 3)].reshape(10000, 32, 32)
            data[:, :, :, 1] = cifar_dict[b'data'][:, int(cifar_dict[b'data'].shape[1] / 3): 2 * int(cifar_dict[b'data'].shape[1] / 3)].reshape(10000, 32, 32)
            data[:, :, :, 2] = cifar_dict[b'data'][:, int(2 * cifar_dict[b'data'].shape[1] / 3):].reshape(10000, 32, 32)
            cifar_data[i * 10000: (i + 1) * 10000, :, :, :] = data
            # append the labels for the current batch to the labels list
            cifar_labels += cifar_dict[b'labels']

        # get the list of the 10 possible image labels (as words)
        cifar_labels_strings = unpickle('./../data/cifar-10-batches-py/batches.meta')[b'label_names']

        # get the label we are interested in
        wanted_label = params.cifar_type

        # get the index of the label we are interested in
        wanted_label_index = None
        for i in range(len(cifar_labels_strings)):
            if cifar_labels_strings[i] == wanted_label:
                wanted_label_index = i

        # get the indices of images with the label wanted
        cifar_labels = np.array(cifar_labels)
        indices = np.where(cifar_labels == wanted_label_index)[0]

        # keep only the images with the label wanted
        cifar_data = cifar_data[indices, :, :, :]
        params.small_images = cifar_data


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    image_h, image_w, image_c = params.image.shape

    # height pt imagine mica
    small_h = params.small_images.shape[1]

    # weight pt imagine mica
    small_w = params.small_images.shape[2]

    # (nr_vertical * small_h) / (nr_horizontal * small_w) = H / W ->
    params.num_pieces_vertical = int(image_h * params.num_pieces_horizontal * small_w / image_w / small_h)

    # redimensioneaza imaginea
    new_h = params.num_pieces_vertical * small_h
    new_w = params.num_pieces_horizontal * small_w
    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon_modified(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
