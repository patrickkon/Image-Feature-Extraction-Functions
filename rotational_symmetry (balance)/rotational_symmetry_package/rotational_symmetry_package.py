"""

Function to call: detectCircles
Recommended parameters: detectCircles(img_path, num_of_centres= 3, threshold=4.1, region=15, radius=[100, 10])

Input Parameter explanation:
img_path = A string representing the path to your image file
num_of_centres = number of rotational centres to be returned
threshold = threshold that will be weighted, to remove low frequency rotational centres
region = Square Area in which to search for a potential rotational centre
radius = maximum and mimimum (respectively) allowed radii value to be considered a rotational centre.
       = Default value for maximum radius is the minimum dimension of input image shape
       = Default value for mimimum radius is pixel value 3

Output value explanation:
Output is a list of arrays.
Each array contains 4 elements (in order):
1.Radius of circular centre
2. X coordinate of circular centre
3. Y coordinate of circular centre
4. Frequency/strength of rotational centre (i.e. number of times it appeared in accumulator array,
   which determines how prevalent it is)


"""

from impy import imarray
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread


# adapted from: https://github.com/PavanGJ/Circle-Hough-Transform/blob/master/main.py#L61

# The functions smooth" and "edge" perform edge detection for the input image
def smoothen(img,display):
    #Using a 3x3 gaussian filter to smoothen the image
    gaussian = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])
    img.load(img.convolve(gaussian))
    if display:
        img.disp
    return img

def edge(img,threshold,display=False):
    #Using a 3x3 Laplacian of Gaussian filter along with sobel to detect the edges
    laplacian = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    #Sobel operator (Orientation = vertical)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    #Generating sobel horizontal edge gradients
    G_x = img.convolve(sobel)

    #Generating sobel vertical edge gradients
    G_y = img.convolve(np.fliplr(sobel).transpose())

    #Computing the gradient magnitude
    G = pow((G_x*G_x + G_y*G_y), 0.5)

    G[G<threshold] = 0
    L = img.convolve(laplacian)
    if L is None:                                                               #Checking if the laplacian mask was convolved
        return
    (M,N) = L.shape

    temp = np.zeros((M+2,N+2))                                                  #Initializing a temporary image along with padding
    temp[1:-1,1:-1] = L                                                         #result hold the laplacian convolved image
    result = np.zeros((M,N))                                                    #Initializing a resultant image along with padding
    for i in range(1,M+1):
        for j in range(1,N+1):
            if temp[i,j]<0:                                                     #Looking for a negative pixel and checking its 8 neighbors
                for x,y in (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1):
                        if temp[i+x,j+y]>0:
                            result[i-1,j-1] = 1                                 #If there is a change in the sign, it is a zero crossing
    img.load(np.array(np.logical_and(result,G),dtype=np.uint8))
    if display:
        img.disp
    return img


def displayCircles(A, img):
    fig = plt.figure()
    plt.imshow(img)
    circleCoordinates = np.argwhere(A)                                          #Extracting the circle information
    circle = []
    for r,x,y in circleCoordinates:
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.show()



def detectCircles(img_path, num_of_centres, threshold, region, radius= None, plot_edge= False, plot_result= False):
    img = imarray.imarray(img_path)
    img = smoothen(img, display=False)
    img = edge(img, 128, display=plot_edge)

    (M,N) = img.shape
    if radius == None:
        R_max = np.max((M,N))
        R_min = 3
    else:
        [R_max,R_min] = radius

    R = R_max - R_min
    #Initializing accumulator array A and B.
    #Accumulator array is a 3 dimensional array with the indicies representing
    #the radius, X coordinate and Y coordinate resectively.
    #Also appending a padding of 2 times R_max to overcome the problems of overflow
    A = np.zeros((R_max,M+2*R_max,N+2*R_max))
    B = np.zeros((R_max,M+2*R_max,N+2*R_max))

    #Precomputing all angles to increase the speed of the algorithm
    theta = np.arange(0,360)*np.pi/180
    edges = np.argwhere(img[:,:])                                               #Extracting all edge coordinates
    for val in range(R):
        r = R_min+val
        #Creating a Circle Blueprint
        bprint = np.zeros((2*(r+1),2*(r+1)))
        (m,n) = (r+1,r+1)                                                       #Finding out the center of the blueprint
        for angle in theta:
            x = int(np.round(r*np.cos(angle)))
            y = int(np.round(r*np.sin(angle)))
            bprint[m+x,n+y] = 1
        constant = np.argwhere(bprint).shape[0]
        for x,y in edges:                                                       #For each edge coordinates
            #Centering the blueprint circle over the edges
            #and updating the accumulator array
            X = [x-m+R_max,x+m+R_max]                                           #Computing the extreme X values
            Y= [y-n+R_max,y+n+R_max]                                            #Computing the extreme Y values
            A[r,X[0]:X[1],Y[0]:Y[1]] += bprint
        A[r][A[r]<threshold*constant/r] = 0


    max_values = []
    max_values_indices = []
    max_value_rot_indicies = []
    for r,x,y in np.argwhere(A):
        # Search for centre of rotations within patches of image A. Area determined by parameter "region"
        temp = A[r-region:r+region,x-region:x+region,y-region:y+region]

        # Maximum number of values to be returned is determined by num_of_centres
        if(temp.size):

            if(len(max_values) < num_of_centres):
                p,a,b = np.unravel_index(np.argmax(temp), temp.shape)
                edit = True
                # Search through values to weed out equivalent circles
                for i in range(0, len(max_values)):
                    if(p+r == max_value_rot_indicies[i][0] + max_values_indices[i][0] and x+a == max_value_rot_indicies[i][1] + max_values_indices[i][1] and y+b == max_value_rot_indicies[i][2] + max_values_indices[i][2]):
                        edit = False
                        break

                if(edit):
                    max_values.append(np.amax(temp))
                    max_values_indices.append((p, a, b))
                    max_value_rot_indicies.append((r, x, y))

            elif(min(max_values) < np.amax(temp)):
                p,a,b = np.unravel_index(np.argmax(temp), temp.shape)
                edit = True

                # Search through values to weed out equivalent circles
                for i in range(0, len(max_values)):
                    if(p+r == max_value_rot_indicies[i][0] + max_values_indices[i][0] and x+a == max_value_rot_indicies[i][1] + max_values_indices[i][1] and y+b == max_value_rot_indicies[i][2] + max_values_indices[i][2]):
                        edit = False
                        break

                if(edit):
                    index = max_values.index(min(max_values))
                    max_values[index] = np.amax(temp)
                    max_values_indices[index] = (p, a, b)
                    max_value_rot_indicies[index] = (r, x, y)

    # Accumulator array used for displaying the circle
    for i in range(len(max_values)):
        B[max_value_rot_indicies[i][0]+(max_values_indices[i][0]-region),max_value_rot_indicies[i][1]+(max_values_indices[i][1]-region),max_value_rot_indicies[i][2]+(max_values_indices[i][2]-region)] = max_values[i]

    if(plot_result):
        r = B[:,R_max:-R_max,R_max:-R_max]
        image_for_rot_display = imread(img_path)
        displayCircles(r, image_for_rot_display)

    # Appending strength (i.e. frequency of appearance in accumulator array "A" for each circular centre)
    # The resulting list contains arrays which have the following elements:
    # Radius, X and Y coordinates of circular centre, and strength
    values = np.argwhere(B)
    values_to_return = []
    for i in range(len(values)):
        values_to_return.append(np.append(values[i], B[values[i][0]][values[i][1]][values[i][2]]))

    return values_to_return

