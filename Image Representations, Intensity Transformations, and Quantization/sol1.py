import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray

GRAYSCALE_REP = 1
RGB_REP = 2
GRAYSCALE_SHAPE = 2
RGB_SHAPE = 3
MAX_VALUE = 256
GRAY_MAX_LEVEL = 255


RGB_TO_YIQ_MATRIX = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ_TO_RGB_MATRIX = np.linalg.inv(RGB_TO_YIQ_MATRIX)


def read_image(filename, representation):
    # reads image from filename returns its array rep

    img = np.float64(imread(filename))/GRAY_MAX_LEVEL

    if representation == GRAYSCALE_REP:
        img = rgb2gray(img)

    return np.float64(img)


def imdisplay(filename, representation):
    # displays image
    img = read_image(filename, representation)
    if representation == GRAYSCALE_REP:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(img)

    plt.show()


def multiply_matrix_by_vector(img, matrix):
    # multiply matrix by vec used for changing from rgb to yiq or yiq to rgb
    a = img[:, :, 0]
    b = img[:, :, 1]
    c = img[:, :, 2]
    new_img = img.copy()
    for i in range(3):
        new_img[:, :, i] = matrix[i][0] * a + matrix[i][1] * b + matrix[i][2] * c

    return new_img


def rgb2yiq(imRGB):
    # return img yiq values
    return multiply_matrix_by_vector(imRGB, RGB_TO_YIQ_MATRIX)


def yiq2rgb(imYIQ):
    # returns img rgb values
    return multiply_matrix_by_vector(imYIQ, YIQ_TO_RGB_MATRIX)


def stretch_img(img_hist):
    # stretching the image

    # cumulative histogram
    cum_h = np.cumsum(img_hist)

    # calculate non zero array
    nonzero_arr = np.nonzero(cum_h)
    # assign min
    min_nonzero = cum_h[nonzero_arr[0][0]]
    # assign max
    max_nonzero = cum_h[nonzero_arr[0][-1]]

    # stretching the image
    im_stretch = np.round(GRAY_MAX_LEVEL * (cum_h - min_nonzero) / (max_nonzero - min_nonzero))

    return im_stretch


def get_y_channel(img,img_yiq):
    # gets y channel from rgb pic
    img_yiq = rgb2yiq(img)
    img = img_yiq[:, :, 0].copy()
    return img, img_yiq


def histogram_equalize(im_orig):
    # applies histogram equalization on im_orig
    img_yiq = []
    was_rgb = False
    img = im_orig.copy()

    # check if img is rgb convert to yiq
    if len(img.shape) == RGB_SHAPE:
        was_rgb = True
        img, img_yiq = get_y_channel(img, img_yiq)

    hist_orig, bins = np.histogram(img, MAX_VALUE)

    # equalizing image
    streched_img = stretch_img(hist_orig)
    equalizing = np.interp(img.flatten(), bins[:-1], streched_img)
    im_eq = equalizing.reshape(img.shape)/GRAY_MAX_LEVEL

    # convert back from yiq to rgb if needed
    if was_rgb:
        img_yiq[:, :, 0] = im_eq
        im_eq = yiq2rgb(img_yiq)

    # equalized histogram
    hist_eq, bins2 = np.histogram(im_eq, MAX_VALUE)

    return [np.float64(im_eq), hist_orig, hist_eq]


def initiate_first_z(img_hist, bins, n_quant):
    # calculate the first z values
    cum_h = np.cumsum(img_hist)

    number_of_pixels_in_segment = int(cum_h[-1] / n_quant)

    # z first setup where z0 is -1
    arr_of_z = np.array([-1] +
                        [bins[np.where(cum_h >= number_of_pixels_in_segment * i)[0][0]] for i in range(1, n_quant)]
                        + [1])
    return arr_of_z


def quantize_algorithm(img, img_hist, bins, z_arr, n_quant, n_iter):
    # applies quantize algorithm to a given img returns equalized image and error array

    arr_of_q = np.zeros(n_quant)
    arr_of_errors = []

    for j in range(n_iter):

        error_sum = 0
        current_z = z_arr.copy()

        for k in range(n_quant):

            # calculate which segment we are working on depending on k
            if k == n_quant - 1:
                segment = np.intersect1d(np.where(bins[:-1] >= z_arr[k])[0], np.where(bins[:-1] <= z_arr[k+1])[0])
            else:
                segment = np.intersect1d(np.where(bins[:-1] >= z_arr[k])[0], np.where(bins[:-1] < z_arr[k+1])[0])

            # check if the sum of the pixels in this segment is larger than zero
            if np.sum(img_hist[segment]) <= 0:
                break

            # calculate q and err using the formula
            arr_of_q[k] = np.dot(bins[segment], img_hist[segment]) / np.sum(img_hist[segment])
            error_sum += np.dot(np.power(arr_of_q[k] - bins[segment], 2), img_hist[segment])

        arr_of_errors.append(error_sum)

        # calculating z using the formula and q values
        z_arr = np.array([0] + [(arr_of_q[m] + arr_of_q[m-1])/2 for m in range(1, n_quant)] + [1])

        # if the z have not changes than the previous ones this means convergence
        if np.array_equal(z_arr, current_z):
            break

    # image quantize
    for seg in range(n_quant):
        seg_index = np.logical_and(img >= z_arr[seg], img < z_arr[seg + 1])
        img[seg_index] = arr_of_q[seg]

    # edge case pixel with 255 intensity
    img[img == 1] = arr_of_q[-1]

    im_qaunt = img
    return [im_qaunt, arr_of_errors]


def quantize(im_orig, n_quant, n_iter):
    # applies quantization on im_orig
    img_yiq = []
    img = im_orig.copy()
    was_rgb = False

    # check if img is rgb convert to yiq
    if len(img.shape) == RGB_SHAPE:
        was_rgb = True
        img, img_yiq = get_y_channel(img, img_yiq)

    orig_hist, bins = np.histogram(img, MAX_VALUE)

    arr_of_z = initiate_first_z(orig_hist, bins, n_quant)

    im_quant, error = quantize_algorithm(img, orig_hist, bins, arr_of_z, n_quant, n_iter)

    # convert back from yiq to rgb if needed
    if was_rgb:
        img_yiq[:, :, 0] = im_quant
        im_quant = yiq2rgb(img_yiq)

    return [np.float64(im_quant), error]
