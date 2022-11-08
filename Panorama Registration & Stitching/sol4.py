# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
import sol4_utils


DERIVATIVE_FILTER = np.array([[1, 0, -1]])
KERNEL_SIZE = 3
RADIUS = 3
FF_DIM = 7


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # Calculate derivatives
    I_x = convolve(im, DERIVATIVE_FILTER)
    I_y = convolve(im, DERIVATIVE_FILTER.T)

    # Blurring
    I_x2 = sol4_utils.blur_spatial(np.square(I_x), KERNEL_SIZE)
    I_y2 = sol4_utils.blur_spatial(np.square(I_y), KERNEL_SIZE)
    I_x_I_y = sol4_utils.blur_spatial(np.multiply(I_x, I_y), KERNEL_SIZE)
    I_y_I_x = sol4_utils.blur_spatial(np.multiply(I_y, I_x), KERNEL_SIZE)

    # calculate matrix M det and trance to find R
    det_M = I_x2 * I_y2 - (I_y_I_x * I_x_I_y)
    trace_M = I_x2 + I_y2
    k = 0.04
    R = det_M - k * np.square(trace_M)

    # locate local maximum points
    local_max = non_maximum_suppression(R)

    result = np.argwhere(local_max.T)
    return result


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = 1 + 2 * desc_rad
    grid = np.indices((K, K))
    descriptor_matrices = []
    for c in pos:
        map_cords = [grid[1] + c[1] - desc_rad, grid[0] + c[0] - desc_rad]
        curr_cord = map_coordinates(im, map_cords, order=1, prefilter=False).T
        # calculate mean and norm to find descriptor matrix
        mean = np.mean(curr_cord)
        norm = np.linalg.norm(curr_cord - mean)
        if norm != 0:
            curr_desc = (curr_cord - mean) / norm
        else:
            curr_desc = np.zeros(curr_cord.shape)
        descriptor_matrices.append(curr_desc)

    return np.array(descriptor_matrices)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    # find keypoints and set for 3 lvl pyramid
    keypoints = spread_out_corners(pyr[0], FF_DIM, FF_DIM, RADIUS)
    pos = np.array(keypoints)/4
    # Calculate descriptor matrices
    descriptor_matrices = sample_descriptor(pyr[2], pos, RADIUS)

    result = [keypoints, descriptor_matrices]
    return result


def find_2nd_max_of_2desc(desc1, desc2):
    """
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :return: 2nd max of each descriptor and score of desc1
    """
    # Reshape descriptors
    num_of_matrices1, n1, m1 = np.array(desc1).shape
    num_of_matrices2, n2, m2 = np.array(desc2).shape
    new_desc1 = np.array(desc1).reshape((num_of_matrices1, n1 * m1))
    new_desc2 = np.array(desc2).reshape((num_of_matrices2, n2 * m2))

    # Calculate match score
    scores = np.dot(new_desc1, new_desc2.T)

    # locate the 2 largest values in each one of the score array
    desc1_2largest = np.argpartition(scores, -2, axis=1)[:, -2:]
    desc2_2largest = np.argpartition(scores.T, -2, axis=1)[:, -2:]

    desc1_frame = scores[np.repeat(np.arange(num_of_matrices1), 2), desc1_2largest.ravel()]
    desc2_frame = scores.T[np.repeat(np.arange(num_of_matrices2), 2), desc2_2largest.ravel()]
    # reshape frame
    desc1_frame = desc1_frame.reshape(num_of_matrices1, 2)
    desc2_frame = desc2_frame.reshape(num_of_matrices2, 2)

    # 2nd max value
    desc1_2nd_max = np.amin(desc1_frame, axis=1)
    desc2_2nd_max = np.amin(desc2_frame, axis=1)
    # reshape
    desc1_2nd_max = desc1_2nd_max.reshape(num_of_matrices1, 1)
    desc2_2nd_max = desc2_2nd_max.reshape(num_of_matrices2, 1)

    return desc1_2nd_max, desc2_2nd_max, scores


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    # locate 2nd max of each desc
    desc1_2ndmax, desc2_2ndmax, scores = find_2nd_max_of_2desc(desc1, desc2)

    # Set properties
    p1 = np.array(scores >= desc1_2ndmax)
    p2 = np.array(scores.T >= desc2_2ndmax)
    p3 = np.array(scores > min_score)

    # create an array of the points which hold the three properties
    result = np.where(p1 & p2.T & p3)
    return result


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    n = pos1.shape[0]

    # Converting points in [x, y] coordinates to homogenous coordinates
    homogenous_cords = np.hstack((pos1, np.ones((n, 1))))

    # multiplying vector from the left with H12 matrix
    multiplied_cords = np.einsum('ij, kj->ki', H12, homogenous_cords)

    # Converting homogenous coordinates to points in [x, y] coordinates
    z_cords = multiplied_cords[:, 2]
    x_cords = np.divide(multiplied_cords[:, 0], z_cords)
    y_cords = np.divide(multiplied_cords[:, 1], z_cords)

    # stack values to get result array
    result = np.vstack((x_cords, y_cords)).T
    return result



def get_matches(points1, points2, p1, p2, inlier_tol, translation_only):
    """
    gets random set of 2 point matches and returns marked matches according to these points
    :param p1: random point
    :param p2: random point
    :param translation_only: given arg
    :return: 2 - elements array includes array of points matching the required eq distance and homography matrix
    """
    # compute homography
    curr_H12 = estimate_rigid_transform(p1, p2, translation_only)
    # apply homography
    curr_applied_homography = apply_homography(points1, curr_H12)
    # compute euclidean distance
    norm = np.linalg.norm(curr_applied_homography - points2, axis=1)
    curr_ed = np.square(norm)
    # mark matches with squared euclidean distance less than inlier_tol
    curr_inlier_matches = np.where(curr_ed < inlier_tol)[0]

    return [curr_inlier_matches, curr_H12]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    inliers_set = []
    n = points1.shape[0]
    for i in range(num_iter):
        # pick random set of 2 point matches
        j = np.random.choice(n, 2)
        # curr points p1j p2j
        p1j = points1[j]
        p2j = points2[j]
        # Calculate homography matrix and mark matches with squared euclidean distance less than inlier_tol
        curr_marked_matches = get_matches(points1, points2, p1j, p2j, inlier_tol, translation_only)[0]
        # update inlier set if num of marked matches in this iter was larger than previous ones
        if len(inliers_set) < len(curr_marked_matches):
            inliers_set = curr_marked_matches

    # recompute homography
    inliers_set = np.array(inliers_set)
    inliers_of_pos1 = points1[inliers_set]
    inliers_of_pos2 = points2[inliers_set]

    marked_matches, H12 = get_matches(points1, points2, inliers_of_pos1, inliers_of_pos2, inlier_tol, translation_only)

    return H12, marked_matches


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x
    ,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    new_im = np.hstack((im1,im2))
    plt.imshow(new_im, cmap='gray')
    m = im1.shape[1]
    num_of_points = points2.shape[0]
    # plot points
    plt.scatter(points1[:, 0], points1[:, 1], c='r', marker='.')
    plt.scatter(points2[:, 0] + m, points2[:, 1], c='r', marker='.')

    for i in range(num_of_points):
        p1 = (points1[i, 0], points2[i, 0] + m)
        p2 = (points1[i, 1], points2[i, 1])
        # plots line (color depends on the condition)
        if i in inliers:

            plt.plot(p1, p2, mfc='r', c='y', lw=.4, ms='3', marker='o')
        else:
            plt.plot(p1, p2, mfc='r', c='b', lw=.5, ms='3', marker='o')

    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    n = len(H_succesive)
    homographies = [1] * (n + 1)

    # in case of i equal to m
    homographies[m] = np.eye(3)

    # in case of i less than m
    for i in range(m - 1, -1, -1):
        homographies[i] = homographies[i + 1].dot(H_succesive[i])
        homographies[i] /= homographies[i][2, 2]

    # in case of i larger than m
    for i in range(m + 1, n + 1):
        inverse_homography = np.linalg.inv(H_succesive[i - 1])
        homographies[i] = homographies[i - 1].dot(inverse_homography)
        homographies[i] /= homographies[i][2, 2]

    return homographies


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    # setup
    top_left = np.array([0, 0])
    bottom_left = np.array([0, h - 1])
    top_right = np.array([w - 1, 0])
    bottom_right = np.array([w - 1, h - 1])
    # apply homography on coordinates
    new_position = apply_homography(np.array([top_left, top_right, bottom_left, bottom_right]), homography)
    # find max
    x_max = np.max(new_position[:, 0])
    y_max = np.max(new_position[:, 1])
    # find min
    x_min = np.min(new_position[:, 0])
    y_min = np.min(new_position[:, 1])
    # set new frame
    frame = np.array([[x_min, y_min], [x_max, y_max]]).astype(np.int)

    return frame


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    im_height, img_width = image.shape
    frame = compute_bounding_box(homography, img_width, im_height)

    x = np.arange(frame[0][0], frame[1][0])
    y = np.arange(frame[0][1], frame[1][1])

    X_coord, Y_coord = np.meshgrid(x, y)
    # reshape X Y coords
    reshaped_X_coord = X_coord.reshape((X_coord.size, 1))
    reshaped_Y_coord = Y_coord.reshape((Y_coord.size, 1))

    coordinates = np.concatenate((reshaped_X_coord, reshaped_Y_coord), axis=1)
    inverse_homography = np.linalg.inv(homography)
    # transform coordinates using homography
    new_coordinates = apply_homography(coordinates, inverse_homography)
    # interpolate coordinates with img coordinates
    interpolated_coordinates = map_coordinates(image, [new_coordinates[:, 1], new_coordinates[:, 0]],
                                               order=1, prefilter=False)
    # reshape
    reshaped_interpolated_coordinates = interpolated_coordinates.reshape((len(y), len(x)))
    return reshaped_interpolated_coordinates


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0,-1]
    for i in range(1, len(homographies)):
        if homographies[i][0,-1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0,-1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2,:2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    local_max[image<(image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:,0], centers[:,1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0,2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
             (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
    ret = corners[legit,:]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]


    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), (panorama * 255).astype(np.uint8))
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))


    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


# testing match features...
#

# def main():
#
#     image = sol4_utils.read_image("ex4-impr-supplementary material/external/oxford1.jpg", 1)
#     image2 = sol4_utils.read_image("ex4-impr-supplementary material/external/oxford2.jpg", 1)
#
#     points1, descs1 = find_features(np.array(sol4_utils.build_gaussian_pyramid(image, 3, 3)[0]))
#     points2, descs2 = find_features(np.array(sol4_utils.build_gaussian_pyramid(image2, 3, 3)[0]))
#     match = match_features(descs1, descs2, 0.8)
#     wow1 = points1[match[0]]
#     wow2 = points2[match[1]]
#     homo, inliers = ransac_homography(wow1, wow2, 17, 49, False)
#     res = display_matches(image, image2, wow1, wow2, inliers)
#
#
# if __name__ == '__main__':
#