'''
Implementation of fast radial symmetry transform in pure Python using OpenCV and numpy.

NOTE: Function to call : frst
      Input image: Greyscale image
      Output array: 6*6 ,64bit floating-point, numpy array

Recommended parameters values:
frst(image, (1,2,3,4,5), 2, 0.2, 2.5, mode='BOTH')

Adapted from:
https://github.com/Xonxt/frst

Which is itself adapted from:
Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for detecting points of interest. Computer Vision, ECCV 2002.
'''

import cv2
import numpy as np


# Calculate vertical gradient for the input image
# 	@param input Input 8-bit image
# 	@param output Output gradient image
def gradx(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use hstack to add back in the columns that were dropped as zeros
    return np.hstack((np.zeros((rows, 1)), (img[:, 2:] - img[:, :-2]) / 2.0, np.zeros((rows, 1))))


# Calculate horizontal gradient for the input image
# 	@param input Input 8-bit image
# 	@param output Output gradient image
def grady(img):
    img = img.astype('int')
    rows, cols = img.shape
    # Use vstack to add back the rows that were dropped as zeros
    return np.vstack((np.zeros((1, cols)), (img[2:, :] - img[:-2, :]) / 2.0, np.zeros((1, cols))))


# Performs fast radial symmetry transform
# img: input image, greyscale
# radii: integer values for 1 or more radius size in pixels (n in the original paper); also used to size gaussian kernel
# alpha: Strictness of symmetry transform (higher=more strict; 2 is good place to start)
# beta: gradient threshold parameter, float in [0,1], recommended value = 0.2, higher = more restrictive
# stdFactor: Standard deviation factor for gaussian kernel, usually kept at n/2 (assumed "n" to be max radius used)
# mode: BRIGHT, DARK, or BOTH, i.e., calculate for only bright symmetric regions, dark symmetric regions, or both
# Output: 6*6 ,64bit floating-point, array

# Example parameter values as recommended in the research paper: frst(image, (1,2,3,4,5), 2, 0.2, 2.5, mode='BOTH')
def frst(img, radii, alpha, beta, stdFactor, mode='BOTH', plot=False):
    # img is resized to 224*224 as indicated in our research paper
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    mode = mode.upper()
    assert mode in ['BRIGHT', 'DARK', 'BOTH']
    dark = (mode == 'DARK' or mode == 'BOTH')
    bright = (mode == 'BRIGHT' or mode == 'BOTH')


    workingDims = img.shape

    # Set up output and M and O working matrices
    O_n = tuple(np.zeros(workingDims, np.int16) for i in range(0,len(radii)))
    M_n = list(np.zeros(workingDims, np.int16) for i in range(0,len(radii)))

    # Calculate gradients
    gx = gradx(img)
    gy = grady(img)

    # Find gradient vector magnitude
    gnorms = np.sqrt(np.add(np.multiply(gx, gx), np.multiply(gy, gy)))

    # Use beta to set threshold - speeds up transform significantly
    # beta*(max of gnorms) is the threshold gradient parameter, anything below that won't be included in Mn, On and so on
    gthresh = np.amax(gnorms) * beta

    # Find x/y distance to affected pixels
    gpx = tuple(np.multiply(np.divide(gx, gnorms, out=np.zeros(gx.shape), where=gnorms != 0), i).round().astype(int) for i in radii)
    gpy = tuple(np.multiply(np.divide(gy, gnorms, out=np.zeros(gy.shape), where=gnorms != 0), i).round().astype(int) for i in radii)

    # Iterate over all pixels (w/ gradient above threshold)
    for coords, gnorm in np.ndenumerate(gnorms):
        if gnorm > gthresh:
            i, j = coords
            # Positively affected pixel
            if bright:
                for index in range(0,len(gpx)):
                    # prevents indexing beyond bounds of O_n and M_n arrays
                    # As manipulating with gpx might cause that to happen
                    if(i + gpx[index][i, j] < workingDims[0] and i + gpx[index][i, j] > 0 and j + gpy[index][i, j] < workingDims[1] and j + gpy[index][i, j] > 0):
                        ppve = (i + gpx[index][i, j], j + gpy[index][i, j])
                        O_n[index][ppve] += 1
                        M_n[index][ppve] += gnorm
            # Negatively affected pixel
            if dark:
                for index in range(0,len(gpx)):
                    # prevents indexing beyond bounds of O_n and M_n arrays
                    # As manipulating with gpy might cause that to happen
                    if(i - gpx[index][i, j] < workingDims[0] and i - gpx[index][i, j] > 0 and j - gpy[index][i, j] < workingDims[1] and j - gpy[index][i, j] > 0):
                        pnve = (i - gpx[index][i, j], j - gpy[index][i, j])
                        O_n[index][pnve] -= 1
                        M_n[index][pnve] -= gnorm



    # Abs and normalize O matrix
    O_n = np.abs(O_n)
    for index in range(0,len(O_n)):
        # Alternative normalization proposed in author's earlier works, mentioned in cited research paper
        O_n[index] = O_n[index] / float(np.amax(O_n[index])/1.5)


    # Normalize M matrix
    # Note: Normalizing O matrix is more important than this M matrix, because O is multiplied by power indicated by "alpha" parameter, later
    for index in range(0, len(M_n)):
        M_max = float(np.amax(np.abs(M_n[index])))
        M_n[index] = M_n[index] / M_max
        M_n[index] = np.nan_to_num(M_n[index])

    # Elementwise multiplication
    F_n = np.multiply(np.power(O_n, alpha), M_n)

    # Gaussian blur
    # As recommended in research paper, assumed "n" to be max radius in radii parameter
    kSize = int(np.ceil(radii[len(radii)-1]))
    kSize = kSize + 1 if kSize % 2 == 0 else kSize

    S = []
    for index in range(0,len(F_n)):
        # Convolves F_n with the specified Gaussian Kernel to get Sn
        S.append(cv2.GaussianBlur(F_n[index], (kSize, kSize), int(radii[len(radii)-1] * stdFactor), int(radii[len(radii)-1] * stdFactor)))

    # Average out the Sn for all radii, to get final result
    S_avg = sum(S) / len(radii)
    S_avg = np.asarray(S_avg)

    # As quoted in research paper (surrounding blur not yet implemented)

    if (plot):
        radial_symmetry_for_display = S_avg
        S_avg_temp = np.nonzero(S_avg == 0)
        # Using linear interpolation to shift the range of array to within 0 -> +255
        radial_symmetry_for_display = np.interp(radial_symmetry_for_display, (radial_symmetry_for_display.min(), radial_symmetry_for_display.max()), (0, +255))

        #According to the paper, elements with value 0 should be grey
        for i in range(0, len(S_avg_temp[0])):
            radial_symmetry_for_display[S_avg_temp[0][i], S_avg_temp[1][i]] = 127

        radial_symmetry_for_display = radial_symmetry_for_display.astype('uint8')
        cv2.imshow("radial", radial_symmetry_for_display)
        cv2.waitKey(0)

    # Output image is resized to 6*6
    # As indicated in the research paper: Principles of Art Features by Sicheng Zhao et al.
    # to quote page 6: "Distribution of symmetry map after radial symmetry transformation"
    S_avg = cv2.resize(S_avg, (6, 6), interpolation=cv2.INTER_CUBIC)

    return S_avg


