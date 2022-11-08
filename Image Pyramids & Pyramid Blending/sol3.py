import numpy as np
from imageio import imread
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os

GRAYSCALE_REP = 1
RGB_REP = 2
GRAY_MAX_LEVEL = 255


def read_image(filename, representation):
    # reads image from filename returns its array rep

    img = np.float64(imread(filename))/GRAY_MAX_LEVEL

    if representation == GRAYSCALE_REP:
        img = rgb2gray(img)

    return np.float64(img)


def create_filter(filter_size):
    # creates gaussian filter with filter_size

    if filter_size == 1:
        return np.array([[1]])

    else:
        filter_vec = np.array([[1, 1]]).astype(np.float64)
        for i in range(filter_size - 2):
            filter_vec = convolve2d(filter_vec, np.array([[1, 1]]))

        return filter_vec / np.sum(filter_vec)


def blur_image(image, filter_vec):
    # blur image with given filter vector

    # convolve img with row vec
    blur_img_rows = convolve(image, filter_vec)
    # convolve im with col vec to get blurred image
    blurred_img = convolve(blur_img_rows, filter_vec.T)

    return blurred_img


def reduce(image, filter_vec):
    # reduce image algorithm

    # blur
    blurred_img = blur_image(image, filter_vec)
    # sub-sample
    result = blurred_img[::2, ::2]
    return result


def expand(image, filter_vec):
    # expand image algorithm

    img_shape = image.shape
    # create zero array with double the size
    expanded_img = np.zeros((2 * img_shape[0], 2 * img_shape[1]))
    # zero padding original im
    expanded_img[::2, ::2] = image
    # blur image
    blurred_img = blur_image(expanded_img, 2 * filter_vec)
    result = blurred_img

    return result


def build_gaussian_pyramid(im, max_levels, filter_size):
    # build gaussian pyramid of image

    pyr = []
    # create filter vector
    filter_vector = create_filter(filter_size)

    # G0
    g_img = im
    pyr.append(g_img)

    # find max level depending on image dimensions
    img_min_dim = min(int(np.log(im.shape[0] // 16) / np.log(2)) + 1,
                      int(np.log(im.shape[1] // 16) / np.log(2)) + 1)
    max_levels = min(max_levels, img_min_dim)

    # build gaussian pyramid
    for i in range(max_levels-1):
        # get next level of gaussian pyramid
        g_img = reduce(np.copy(g_img), filter_vector)
        # append G_i(reduced) to pyramid
        pyr.append(g_img)

    return pyr, filter_vector


def build_laplacian_pyramid(im, max_levels, filter_size):
    # build laplacian pyramid of image

    pyr = []
    # build gaussian pyramid and create filter
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    # build laplacian pyramid
    for i in range(len(gaussian_pyramid) - 1):
        # get next level of laplacian pyramid
        l_img = gaussian_pyramid[i] - expand(np.copy(gaussian_pyramid[i + 1]), filter_vec)
        # append L_i to pyramid
        pyr.append(l_img)

    pyr.append(gaussian_pyramid[-1])

    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    # rebuild image using laplacian pyramid

    img = lpyr[-1] * coeff[-1]

    for i in range(len(lpyr) - 2, -1, -1):
        img = lpyr[i] * coeff[i] + expand(img, filter_vec)

    return img


def render_pyramid(pyr, levels):
    # rendering pyramid

    # find min value
    levels = min(len(pyr),levels)

    start_column_index = 0

    # setup
    img_shape = pyr[0].shape[0]
    img_shape1 = int(pyr[0].shape[1] * (1 - np.power(0.5, levels)) / 0.5)
    res = np.zeros((img_shape, img_shape1))

    for lvl in range(levels):
        img = np.copy(pyr[lvl])
        # stretching values
        min_value, max_value = np.min(img), np.max(img)
        stretched_img = np.round(255 * (img - min_value) / (max_value - min_value))
        curr_shape = stretched_img.shape
        # add to result
        res[:curr_shape[0], start_column_index:curr_shape[1] + start_column_index] = stretched_img
        # update index
        start_column_index += curr_shape[1]

    return res


def display_pyramid(pyr, levels):
    # renders and displays image pyramid

    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    # blending two images using a given mask

    # building pyramid for each img
    img1_pyramid = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0]
    img2_pyramid = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_pyramid, filter_vec = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)

    # blending images using give algorithm
    mask_img1_multiply = np.multiply(mask_pyramid, img1_pyramid)
    mask_img2_multiply = np.multiply(np.subtract(1, mask_pyramid), img2_pyramid)
    final_result = mask_img1_multiply + mask_img2_multiply

    coeff = np.ones(final_result.shape[0])

    return np.clip(laplacian_to_image(final_result, filter_vec, coeff), 0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_rgb_images(im1, im2, mask):
    # blending rgb images

    blended_img = np.zeros(im1.shape)
    blended_img[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 10, 5, 5)
    blended_img[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 10, 5, 5)
    blended_img[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 10, 5, 5)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(blended_img)
    plt.show()

    return im1, im2, mask, blended_img


def blending_example1():
    # examples

    im1 = read_image(relpath("solarvynil.jpg"), 2)
    im2 = read_image(relpath("solars.jpg"), 2)
    mask = np.round(read_image(relpath("solarvynilmask.jpg"), 1)).astype(np.bool)

    result1 = blending_rgb_images(im1, im2, mask)
    return result1

def blending_example2():
    # examples

    im1 = read_image(relpath("waterfall.jpg"), 2)
    im2 = read_image(relpath("electricstairs.jpg"), 2)
    mask = np.round(read_image(relpath("waterfallmask.jpg"), 1)).astype(np.bool)

    result2 = blending_rgb_images(im1, im2, mask)
    return result2
