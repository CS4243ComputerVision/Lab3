import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
import math

from utils import pad, get_output_space, unpad

import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def warp_image(src, dst, h_matrix):
    dst = dst.copy()
    dst = cv2.warpPerspective(dst, np.linalg.inv(h_matrix), (src.shape[1] + dst.shape[1], src.shape[0]))
    dst[0:src.shape[0], 0:src.shape[1]] = src
    return dst

def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas

def transform_homography(src, h_matrix, getNormalized = True):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    """
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)

    return transformed


# def normalize(points):
#     mean = np.mean(points)
#     points = points - mean
#     scale = math.sqrt(2)/np.linalg.norm(points)
#     return points * scale, mean, scale



# def compute_homography(src, dst):
#     """Calculates the perspective transform from at least 4 points of
#     corresponding points using the **Normalized** Direct Linear Transformation
#     method.
#     Args:
#         src (np.ndarray): Coordinates of points in the first image (N,2)
#         dst (np.ndarray): Corresponding coordinates of points in the second
#                           image (N,2)
#     Returns:
#         h_matrix (np.ndarray): The required 3x3 transformation matrix H.
#     Prohibited functions:
#         cv2.findHomography(), cv2.getPerspectiveTransform(),
#         np.linalg.solve(), np.linalg.lstsq()
#     """
#     h_matrix = np.eye(3, dtype=np.float64)

#     ### YOUR CODE HERE

#     N = src.shape[0]

#     # todo: normalisation
#     X1, mean1, scale1 = normalize(src)
#     X2, mean2, scale2 = normalize(dst)

#     # X1 = src
#     # X2 = dst

#     A = []
#     for i in range(N):
#         y1, x1 = X1[i]
#         y2, x2 = X2[i]
#         w1 = w2 = 1
#         A.append([0,0,0,-x1*w2,-y1*w2,-w1*w2,x1*y2,y1*y2,w1*y2])
#         A.append([x1*w2,y1*w2,w1*w2,0,0,0,-x1*x2,-y1*x2,-w1*x2])

#     #todo: why is this taking forever
#     u, s, vh = np.linalg.svd(np.array(A))

#     # h_matrix = s.reshape(3, 3)
#     # s = s/np.linalg.norm(s)
#     # h_matrix = ((s.reshape(3, 3) / scale2 + mean2) - mean1) * scale1

#     ### END YOUR CODE

#     return h_matrix



# With Reference to:
# (1) https://stackoverflow.com/questions/50595893/dlt-vs-homography-estimation
# Summary:
# A little math shows that a good choice for the m terms are the averages of the x,y coordinates in each set of image points, and for the s terms you use the standard deviations (or twice the standard deviation) times 1/sqrt(2).
# You can express this normalizing transformation in matrix form: q = T p, where T = [[1/sx, 0, -mx/sx], [0, 1/sy, -my/sy], [0, 0, 1]], and likewise q' = T' p'.

def normalise(points):
    mean = np.mean(points, axis=0)
    scale = 2 * np.std(points) / math.sqrt(2)

    T = np.array([[1/scale, 0, -mean[0]/scale], [0, 1/scale, -mean[1]/scale], [0, 0, 1]])

    homogenous_points = np.insert(points, 2, values=1, axis=1)

    normalised_points = homogenous_points.dot(T.T) # function homogenises the given points i think

    return normalised_points, T

# With Reference to: 
# (1) (Refer this for step 3)https://www.mail-archive.com/floatcanvas@mithis.com/msg00513.html
# (2) (Refer Slide 17 for big picture on the overall steps to be performed) https://web.archive.org/web/20150929063658/http://www.ele.puc-rio.br/~visao/Topicos/Homographies.pdf
# (3) Refer CS4243 slides for DLT computation
def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """
    h_matrix = np.eye(3, dtype=np.float64)

    ### YOUR CODE HERE

    # Step 1: Perform Normalisation
    X1, T1 = normalise(src)
    X2, T2 = normalise(dst)

    N = np.array(X1).shape[0]

    # Step 2: Apply DLT on normalised points
    A = []
    for i in range(N):
        x1, y1, w1 = X1[i]
        x2, y2, w2 = X2[i]
        A.append([0,0,0,-x1*w2,-y1*w2,-w1*w2,x1*y2,y1*y2,w1*y2])
        A.append([x1*w2,y1*w2,w1*w2,0,0,0,-x1*x2,-y1*x2,-w1*x2])

    u, s, vh = np.linalg.svd(np.array(A))

    # Step 3: Compute H_Matrix on normalised points
    # The parameters are in the last line of Vh and we need to normalise them:
    L = vh[-1,:]
    h_matrix_normalised = L.reshape(3, 3)

    # Step 4: Denormalise H_Matrix_Normalised by computing (inverse(T).H_Matrix_Normalised. T_Prime) 
    T2I = np.linalg.inv(T2)
    h_matrix = T2I @ h_matrix_normalised @ T1

    ### END YOUR CODE

    return h_matrix



def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    ### YOUR CODE HERE
    # Compute x-y derivatives of image
    Ix = filters.sobel_v(img)
    Iy = filters.sobel_h(img)

    # Compute product of derivative at every pixel
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Compute sum of the product of derivative at every pixel
    Sxx = convolve(Ixx, window)
    Syy = convolve(Iyy, window)
    Sxy = convolve(Ixy, window)

    # Get response
    for h in range(0,H):
        for w in range(0, W):
            response[h][w] = (Sxx[h][w]*Syy[h][w] - Sxy[h][w]*Sxy[h][w]) - (k * math.pow(Sxx[h][w] + Syy[h][w], 2))

    ### END YOUR CODE
    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    ### YOUR CODE HERE
    standard_deviation = np.std(patch)
    mean = np.mean(patch)

    if standard_deviation > 0.0:
        feature = (patch - mean) / standard_deviation
    else:
        feature = patch - mean

    ### END YOUR CODE
    return feature.flatten()


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    top_ind = np.argpartition(dists, 1)[:, :2]
    dist_ratio = dists[np.arange(N), top_ind[:,0]] / dists[np.arange(N), top_ind[:,1]]
    mask = dist_ratio < threshold
    accepted_row = np.argwhere(mask)
    accepted_col = top_ind[mask]

    matches = np.array([[accepted_row[i, 0], accepted_col[i, 0]] for i in range(len(accepted_col))])
    ### END YOUR CODE

    return matches

def ransac(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

#     # Please note that coordinates are in the format (y, x)
#     matched1 = pad(keypoints1[matches[:,0]])
#     matched2 = pad(keypoints2[matches[:,1]])
#     matched1_unpad = keypoints1[matches[:,0]]
#     matched2_unpad = keypoints2[matches[:,1]]
    
    # Extract matched keypoints
    p1 = keypoints1[matches[:,0]]
    p2 = keypoints2[matches[:,1]]

    # For p1 and p2 the x-coordinate is at p1[:,1] and y at p1[:,0]. Swapping the columns for consistency.
    p1[:, [0,1]] = p1[:, [1,0]]
    p2[:, [0,1]] = p2[:, [1,0]]
    
    # Please note that coordinates are in the format (y, x)
    matched1 = pad(p1)
    matched2 = pad(p2)
    matched1_unpad = p1
    matched2_unpad = p2

    max_inliers = np.zeros(N)
    n_inliers = 0
#     new_src = []
#     new_dst = []

    # RANSAC iteration start
    for i in range(n_iters):
        temp_max = np.zeros(N, dtype=np.int32)
        temp_n = 0
        idx = np.random.choice(N, n_samples, replace=False)
        p1_unpad = matched1_unpad[idx, :]
        p2_unpad = matched2_unpad[idx, :]
        H = compute_homography(p2_unpad, p1_unpad)
        transformed = transform_homography(matched2_unpad, H)
        temp_max = np.linalg.norm(transformed - matched1_unpad, axis=1) ** 2 < threshold
        temp_n = np.sum(temp_max)
        if temp_n > n_inliers:
            max_inliers = temp_max.copy()
            n_inliers = temp_n
    H = compute_homography(matched2_unpad[max_inliers], matched1_unpad[max_inliers])

    ### YOUR CODE HERE
    
#     for i in range(n_iters):
#         # Select a random set of n_samples of matches
#         random_idx = np.random.choice(N, n_samples, replace=False)       

#         # Compute homography matrix
#         # src (np.ndarray): Coordinates of points in the first image (N,2)
#         # dst (np.ndarray): Corresponding coordinates of points in the second image (N,2)
#         src = matched1_unpad[random_idx, :]
#         dst = matched2_unpad[random_idx, :]
#         h = compute_homography(src, dst)        

#         # Compute sum of squared difference & Find inliers using provided threshold
#         transformed = transform_homography(src, h)
#         sum_squared_difference = 0
        
#         inliers = []
#         src_inliers = []
#         dst_inliers = []
#         for idx, point in enumerate(transformed):            
#             squared_difference = np.sum((point - dst[idx]) ** 2)
#             sum_squared_difference += squared_difference
#             if squared_difference < threshold:
#                 inliers.append(idx)
#                 src_inliers.append(src[idx])
#                 dst_inliers.append(dst[idx])
        
#         if len(inliers) > n_inliers:
#             n_inliers = len(inliers)
#             max_inliers = inliers
#             new_src = src_inliers
#             new_dst = dst_inliers

#     # Recomputer least squares estimate using only the inliers
#     print(n_inliers)
#     print(max_inliers)
#     H = compute_homography(new_src, new_dst)
    
    ### END YOUR CODE
    return H, matches[max_inliers]


def sift_descriptor(patch):
    """
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """

    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4,4,8))

    ### YOUR CODE HERE
    # https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
    
    # Calculate the orientation of each pixel in the patch, split into 8 bins
    hist_map = []
    mag_map = []
    for i, row in enumerate(patch):
        hist_map_row = []
        mag_map_row = []
        for j in range(len(row)):
            grad_x = dx[i, j]
            grad_y = dy[i, j]
            grad_magnitude = math.sqrt(grad_x * grad_x + grad_y * grad_y)
            grad_orientation = math.degrees(math.atan2(grad_y, grad_x))
            hist_index = int(math.floor(grad_orientation * 8 / 360)) + 4
            hist_orientation = hist_index % 8
            hist_map_row.append(hist_orientation)
            mag_map_row.append(grad_magnitude)
        hist_map.append(hist_map_row)
        mag_map.append(mag_map_row)
    hist_map = np.array(hist_map)
    mag_map = np.array(mag_map)
    #print(hist_map)
    #print(mag_map)
    
    # Divide patch into 4x4 grid cells of length 4, Compute the histogram for each cell
    columns = np.split(hist_map, 4,axis=1)
    mag_columns = np.split(mag_map, 4, axis=1)
    for i, (column, mag_column) in enumerate(zip(columns, mag_columns)):
        rows = np.split(column, 4, axis=0)
        mag_rows = np.split(mag_column, 4, axis=0)
        for j, (arr, mag_arr) in enumerate(zip(rows,mag_rows)):
            arr = arr.flatten()
            mag_arr = mag_arr.flatten()
            for element, mag_element in zip(arr, mag_arr):
                histogram[i,j,element] += mag_element
    #print(histogram)
    
    # Normalise histogram
    for i, row in enumerate(histogram):
        for j, column in enumerate(row):
            sub_hist = histogram[i, j]
            histogram[i, j] = np.linalg.norm(sub_hist)     
    #print(histogram)
    
    # Append histograms into 128 dimensional vector
    feature = histogram.flatten()
    #print(feature)
    
    # END YOUR CODE

    return feature


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    raise NotImplementedError() # Delete this line
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    raise NotImplementedError() # Delete this line
    ### END YOUR CODE

    return panorama
