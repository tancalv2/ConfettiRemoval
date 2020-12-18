#  ScatterInpaintingAlgorithm for Confetti Removal: 

The algorithm developed is based on course material and 'Scatter Inpainting Algorithm for Rain or Snow Removal in a Single Image' [url](https://ieeexplore.ieee.org/document/8781043/)

---

There are some adjustable parameters for this method

- Gaussian Kernel radius and sigma values (K, SIG)
- Image Bounding Box Padding (PAD)
- Confetti min/max sizes (MIN_SIZE, MAX_SIZE)
- 4 or 8 nearest neighbour direction (DIR_4PX, default = True)
- plot individual bounding box (PLOT_BOX, default = False)

---

## Results

The following is the timelapse of the confetti removal process:

![img_clr](https://github.com/tancalv2/ConfettiRemoval/blob/main/ScatterInpaintingAlgorithm/res/videos/img_clr_4x.gif)
![img_edge](https://github.com/tancalv2/ConfettiRemoval/blob/main/ScatterInpaintingAlgorithm/res/videos/img_edge_4x.gif)
