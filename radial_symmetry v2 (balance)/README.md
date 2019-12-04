# Feature - Radial Symmetry (Balance)
Folder information: <br />
Images - For your viewing. Contains input and output images that were tested <br /><br />
radial_symmetry_package - contains the actual package that can be directly imported
<br />

Usage: Simply import radial_symmetry_v2_package
Usage guide: 
1. Copy folder "radial_symmetry_v2_package" to your directory
2. In python file: from radial_symmetry_v2_package import radial_symmetry_balance
<br /> <br />
Note: **frst** is the function that computes radial symmetry

Note: recommended parameter values (as indicated in research paper)

frst(image, (1,2,3,4,5), 2, 0.2, 2.5, mode='BOTH')

* Note that the second argument accepts a tuple, so for single radius, use: (5,)

<br />
For more explanation on parameter meaning/values:
<br />
1. Peek into frst function definition when calling
<br />
2. Take a look at the research paper
<br /><br /><br />


#### Version 2 Changes:
In general, modified to conform as closely as I could with the research paper

1. O_n and M_n arrays have same dimensions as input image
2. Enable multi radii symmetry measurement
3. Gaussian Blur, with standard deviation applied to both X and Y dimensions
4. Linear interpolation to shift range of output array S, into range 0 -> +255 (to "conform" more to research paper output image results)


#### Potential differences between my version and research paper approach:
1. Did not use Sobel operator for gradient calculation
2. Did not follow "experimentation based" approach to detecting normalization factor "kn", which is used in line 112. Instead, used alternative approach which is in fact author's earlier work, as indicated in cited research paper.
3. Assumed that the average of the symmetry contributions "S", is averaged by summing up the individual radii Sn, and dividing by number of radii used (if using recommended parameters, then =5)
4. Assumed that the Gaussian blur kernel size, and standard deviation factor recommended values (as shown in page 8 of cited research paper), with values "n", and "n/2" respectively, have "n" that is referring to the LARGEST radius used in the parameter list.



##### Based on this research paper: Loy, G., & Zelinsky, A. (2002). A Fast Radial Symmetry Transform for Detecting Points of Interest. ECCV.



