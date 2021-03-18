from parameters import *
import numpy as np
import pdb
import timeit


def check_neighbours(params: Parameters, i, j, image_index, pieces_used):
    '''
    :param params: parameters of the program
    :param i: line of current patch in the mosaic image
    :param j: column of current patch in the mosaic image
    :param image_index: index of the small-image that we would like to place on the current patch
    :param pieces_used: matrix in which we store the indexes of small images placed in the mosaic
    :return:
    '''
    # check north
    if i > 0 and pieces_used[i - 1][j] == image_index:
        return False

    # check south
    if i < params.num_pieces_vertical - 1 and pieces_used[i + 1][j] == image_index:
        return False

    # check west
    if j > 0 and pieces_used[i][j - 1] == image_index:
        return False

    # check east
    if j < params.num_pieces_horizontal - 1 and pieces_used[i][j + 1] == image_index:
        return False

    return True


def check_neighbours_hexa(params: Parameters, i, j, image_index, pieces_indexes):
    '''
    :param params: parameters of the program
    :param i: index of line of the current hexagon
    :param j: index of column of the current hexagon
    :param image_index: index of a small-image that we want to use in the mosaic
    :param pieces_indexes: matrix in which we store the indexes of small-images used in the mosaic
    :return: True if the small-image that we want to use wasn't already placed on any of the neighbours of the current
    hexagon
    '''
    num_pieces_vertical = pieces_indexes.shape[0]
    num_pieces_horizontal = pieces_indexes.shape[1]

    # check north
    if i > 1 and pieces_indexes[i - 2][j] == image_index:
        return False

    # check south
    if i < num_pieces_vertical - 2 and pieces_indexes[i + 2][j] == image_index:
        return False

    # check north-west
    if i > 0 and j > 0 and pieces_indexes[i - 1][j - 1] == image_index:
        return False

    # check south-west
    if i < num_pieces_vertical - 1 and j > 0 and pieces_indexes[i + 1][j - 1] == image_index:
        return False

    # check north-east
    if i > 0 and j < num_pieces_horizontal - 1 and pieces_indexes[i - 1][j + 1] == image_index:
        return False

    # check south-east
    if i < num_pieces_vertical - 1 and j < num_pieces_horizontal - 1 \
            and pieces_indexes[i + 1][j + 1] == image_index:
        return False

    return True


def add_pieces_grid_modified(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    # record index of each piece used in the mosaic
    pieces_used = np.full((params.num_pieces_vertical, params.num_pieces_horizontal), -1)

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # select current patch in the resized image
                current_patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :]

                # record distance between the current patch and each small image
                distances = list(np.zeros(params.small_images.shape[0]))

                for index in range(params.small_images.shape[0]):
                    patch_means = np.mean(current_patch, axis=(0, 1))
                    small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                    distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

                # get the index of the small-image with the smallest difference from the current patch
                index_min = np.argmin(distances)

                # if the small image with smallest difference from the patch doesn't fit go to the next one and so on
                while not check_neighbours(params, i, j, index_min, pieces_used):
                    distances[index_min] = np.max(distances) + 1
                    index_min = np.argmin(distances)

                # record the index of the small-image used in order to be assured that this patch's future neighbours
                # will have different small-images assigned to them
                pieces_used[i][j] = index_min

                # put the small-image on the current patch
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index_min]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    # print(params.num_pieces_vertical)
    # print(params.num_pieces_horizontal)

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=N, size=1)
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                # select current patch in the resized image
                current_patch = params.image_resized[i * H: (i + 1) * H, j * W: (j + 1) * W, :].copy()

                # record distance between the current patch and each small image
                distances = np.zeros(params.small_images.shape[0])

                for index in range(params.small_images.shape[0]):
                    patch_means = np.mean(current_patch, axis=(0, 1))
                    small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                    distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

                # get the index of the small-image with the smallest difference from the current patch
                index_min = np.argmin(distances)

                # put the small-image on the current patch
                img_mosaic[i * H: (i + 1) * H, j * W: (j + 1) * W, :] = params.small_images[index_min]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))
    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def check_if_complete(img_track):
    # count total number of pixels in the resized image
    total_pixels = img_track.shape[0] * img_track.shape[1]
    # if all pixels have been given a value the mosaic is complete
    pixels_set = np.sum(img_track)

    if pixels_set == total_pixels:
        return True
    print('Building mosaic %.2f%% ' % (100 * (pixels_set / total_pixels)), end='')
    print('Pixels set: {} / {}'.format(pixels_set, total_pixels))
    return False


def add_pieces_random(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    img_track = np.zeros((params.image_resized.shape[0], params.image_resized.shape[1]))

    # list of coordinates for each posible top-left corner of a patch in the resized image
    indices_list = [(i, j) for i in range(0, h - H, int(H/5)) for j in range(0, w - W, int(W/5))]
    # indices_list = [(i, j) for i in range(0, h - H) for j in range(0, w - W)]

    for i in range(0, h - H, int(H/5)):
        indices_list.append((i, w - W))

    for j in range(0, w - W, int(W/5)):
        indices_list.append((h - H, j))

    indices_list.append((h - H, w - W))

    while True:
        # from time to time, check if the mosaic is complete
        coin = np.random.randint(0, 1000, 1)[0]
        if coin < 8:
            if check_if_complete(img_track):
                break

        # pick a random top left corner for the current patch (not used corner)
        if len(indices_list) == 0:
            break
        index = np.random.randint(0, len(indices_list), 1)[0]
        left_i = indices_list[index][0]
        left_j = indices_list[index][1]

        # remove left corner used (so it won't be chosen again)
        indices_list.pop(index)

        # select current patch in the resized image
        current_patch = params.image_resized[left_i: left_i + H, left_j: left_j + W, :]

        # record distance between the current patch and each small image
        distances = np.zeros(params.small_images.shape[0])
        for index, img in enumerate(params.small_images):
            patch_means = np.mean(current_patch, axis=(0, 1))
            small_image_means = np.mean(params.small_images[index], axis=(0, 1))
            distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

        index_min = np.argmin(distances)
        img_mosaic[left_i: left_i + H, left_j: left_j + W, :] = params.small_images[index_min]

        # keep track of pixels with value set
        img_track[left_i: left_i + H, left_j: left_j + W] = 1

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_hexagon(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    # add black pixels to the resized image for padding
    big_image = np.zeros((h + H, w + W + int(1/3 * W), C))
    big_image[int(H/2): h + int(H/2), int(W/2): w + int(W/2), :] = params.image_resized

    print(img_mosaic.shape)
    print(big_image.shape)
    first_row_start = int(H / 2)

    # build mask
    mask = np.ones((H, W, C))

    # fill top-left and top-right corners with 0's
    for i in range(int(H / 2)):
        for j in range(0, int(np.floor(W/3)) - i):
            mask[i, j, :] = 0
            mask[i, W - j - 1, :] = 0

    # fill bottom-left and bottom-right corners with 0's
    for i in range(H - 1, int(H / 2) - 1, -1):
        for j in range(0, int(np.floor(W/3)) - (H - 1 - i)):
            mask[i, j, :] = 0
            mask[i, W - j - 1, :] = 0

    # fill the mosaic with the first type of hexagons
    for i in range(first_row_start, big_image.shape[0] - H, int(H)):
        for j in range(0, big_image.shape[1] - W, int(4/3 * W)):
            current_patch = big_image[i: i + H, j: j + W, :]

            # record distance between the current patch and each small image
            distances = np.zeros(params.small_images.shape[0])

            for index in range(params.small_images.shape[0]):
                patch_means = np.mean(current_patch, axis=(0, 1))
                small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

            index_min = np.argmin(distances)

            big_image[i: i + H, j: j + W, :] = (1 - mask) * big_image[i: i + H, j: j + W, :] \
                                                + mask * params.small_images[index_min]
        print("First step rows done: {} / {}".format((i - first_row_start) / H, (big_image.shape[0] - 2 * H) / H))

    # fill the mosaic with the second type of hexagons
    for i in range(0, big_image.shape[0] - H + 1, H):
        for j in range(int(2/3 * W), big_image.shape[1] - W + 1, int(4 / 3 * W)):
            current_patch = big_image[i: i + H, j: j + W, :]

            # record distance between the current patch and each small image
            distances = np.zeros(params.small_images.shape[0])

            for index in range(params.small_images.shape[0]):
                patch_means = np.mean(current_patch, axis=(0, 1))
                small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

            index_min = np.argmin(distances)

            big_image[i: i + H, j: j + W, :] = (1 - mask) * big_image[i: i + H, j: j + W, :] \
                                                + mask * params.small_images[index_min]
        print("Second step rows done: {} / {}".format(i / H, (big_image.shape[0] - H) / H))

    img_mosaic = big_image[H//2: h + H//2, W//2: w + W//2, :]

    # return big_image
    return img_mosaic


def add_pieces_hexagon_modified(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    N, H, W, C = params.small_images.shape
    h, w, c = params.image_resized.shape
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    # add black pixels to the resized image for padding
    big_image = np.zeros((h + H, w + W + int(1/3 * W), C))
    big_image[int(H/2): h + int(H/2), int(W/2): w + int(W/2), :] = params.image_resized

    # keep track of pieces put into the mosaic
    pieces_indexes = np.full((2 * params.num_pieces_vertical + 1, 2 * int(big_image.shape[1] / (4/3 * W)) + 1), -1)

    print(img_mosaic.shape)
    print(big_image.shape)
    first_row_start = int(H / 2)

    # build mask
    mask = np.ones((H, W, C))

    # fill top-left and top-right corners with 0's
    for i in range(int(H / 2)):
        for j in range(0, int(np.floor(W/3)) - i):
            mask[i, j, :] = 0
            mask[i, W - j - 1, :] = 0

    # fill bottom-left and bottom-right corners with 0's
    for i in range(H - 1, int(H / 2) - 1, -1):
        for j in range(0, int(np.floor(W/3)) - (H - 1 - i)):
            mask[i, j, :] = 0
            mask[i, W - j - 1, :] = 0

    row_index = 1

    # fill the mosaic with the first type of hexagons
    for i in range(first_row_start, big_image.shape[0] - H, int(H)):
        col_index = 0
        for j in range(0, big_image.shape[1] - W, int(4/3 * W)):
            current_patch = big_image[i: i + H, j: j + W, :]

            # record distance between the current patch and each small image
            distances = np.zeros(params.small_images.shape[0])

            for index in range(params.small_images.shape[0]):
                patch_means = np.mean(current_patch, axis=(0, 1))
                small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

            index_min = np.argmin(distances)

            # if the small image with smallest difference from the patch doesn't fit go to the next one and so on
            while not check_neighbours_hexa(params, row_index, col_index, index_min, pieces_indexes):
                distances[index_min] = np.max(distances) + 1
                index_min = np.argmin(distances)

            pieces_indexes[row_index][col_index] = index_min

            big_image[i: i + H, j: j + W, :] = (1 - mask) * big_image[i: i + H, j: j + W, :] \
                                                + mask * params.small_images[index_min]
            col_index += 2
        row_index += 2
        print("First step rows done: {} / {}".format((i - first_row_start) / H, (big_image.shape[0] - 2 * H) / H))

    row_index = 0

    # fill the mosaic with the second type of hexagons
    for i in range(0, big_image.shape[0] - H + 1, H):
        col_index = 1
        for j in range(int(2/3 * W), big_image.shape[1] - W + 1, int(4 / 3 * W)):
            current_patch = big_image[i: i + H, j: j + W, :]

            # record distance between the current patch and each small image
            distances = np.zeros(params.small_images.shape[0])

            for index in range(params.small_images.shape[0]):
                patch_means = np.mean(current_patch, axis=(0, 1))
                small_image_means = np.mean(params.small_images[index], axis=(0, 1))
                distances[index] = np.sqrt(np.sum((patch_means - small_image_means) ** 2))

            index_min = np.argmin(distances)

            # if the small image with smallest difference from the patch doesn't fit go to the next one and so on
            while not check_neighbours_hexa(params, row_index, col_index, index_min, pieces_indexes):
                distances[index_min] = np.max(distances) + 1
                index_min = np.argmin(distances)

            pieces_indexes[row_index][col_index] = index_min

            big_image[i: i + H, j: j + W, :] = (1 - mask) * big_image[i: i + H, j: j + W, :] \
                                                + mask * params.small_images[index_min]
            col_index += 2
        row_index += 2
        print("Second step rows done: {} / {}".format(i / H, (big_image.shape[0] - H) / H))

    img_mosaic = big_image[H//2: h + H//2, W//2: w + W//2, :]

    # return big_image
    return img_mosaic
