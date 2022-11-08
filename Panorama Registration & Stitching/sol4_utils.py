from scipy.signal import convolve2d
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

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


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img
