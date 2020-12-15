#  ScatterInpaintingAlgorithm for Confetti Removal: 

Algorithm developed based on course material and 'Scatter Inpainting Algorithm for Rain or Snow Removal in a Single Image' [url](https://ieeexplore.ieee.org/document/8781043/)

---

# ADJUSTABLE PARAMETERS

## LOAD IMAGE - change image file
img = cv2.imread('img0036.png', 0)
img_clr = cv2.imread('img0036.png')

## Gaussian Kernel and sigma values
K = 3
SIG = 1

## Image Bounding Box Padding
PAD = 5

## confetti min/max sizes
MIN_SIZE = 5
MAX_SIZE = 50

## direction, either 4 or 8 nearest neighbour: T/F - default: True
DIR_4PX = True

## plot individual bounding box - either T/F, default: False
PLOT_BOX = False