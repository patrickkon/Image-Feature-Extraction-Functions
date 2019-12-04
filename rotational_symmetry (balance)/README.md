# Feature - Rotational Symmetry (Balance)
Folder information: <br />
images (num_of_centres =3)- For your viewing. Contains input and output images that were tested, with parameter num_of_centres set to 3 <br /><br />
rotational_symmetry_package - contains the actual package that can be directly imported
<br />

Usage: </br>
##### from rotational_symmetry_package import rotational_symmetry_package
<br /> <br />
Note: **detectCircles** is the function that computes radial symmetry

Note: Recommended parameter values

rotational_symmetry_package.detectCircles(img_path, num_of_centres= 3, threshold=4.1, region=15, radius=[100, 10])



Input Parameter explanation: </br>
img_path = A string representing the path to your image file </br>
num_of_centres = number of rotational centres to be returned </br>
threshold = threshold that will be weighted, to remove low frequency rotational centres </br>
region = Square Area in which to search for a potential rotational centre </br>
radius = maximum and mimimum (respectively) allowed radii value to be considered a rotational centre. </br>
       = Default value for maximum radius is the minimum dimension of input image shape </br>
       = Default value for mimimum radius is pixel value 3 </br>

Output value explanation: </br>
Output is a list of arrays. </br>
Each array contains 4 elements (in order): </br>
1.Radius of circular centre </br>
2. X coordinate of circular centre </br>
3. Y coordinate of circular centre </br>
4. Frequency/strength of rotational centre (i.e. number of times it appeared in accumulator array, </br>
   which determines how prevalent it is) </br>


<br />
For more explanation on parameter meaning/values:
<br />
1. Peek into detectCircles function definition when calling
<br />
2. Take a look at the research paper
<br /><br /><br />



##### Based on this research paper: Loy, G., & Eklundh, J. (2006). Detecting Symmetry and Symmetric Constellations of Features. ECCV.



