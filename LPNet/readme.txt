This is a implementation of our paper [1] and for non-commercial use only. 

You need to install Python with Tensorflow-GPU (1.12.0 or higher version) to run this code.



Usage:


1. Preparing training data: put rainy images into "/TrainData/input" and label images into "/TrainData/label". Note that the pair images' indexes **must be** the same.

2. Run 
"training.py" for training and trained models should be generated at "/model".

3. After training, run 
"testing.py" to test new images, the reuslts should be generated at "/test_img/results".

4. We also release our real-world rainy image dataset at:  https://xueyangfu.github.io/projects/LPNet.html


If this code and dataset help your research, please cite our paper:

[1] X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. "Lightweight Pyramid Networks for Image Deraining", IEEE Transactions on Neural Networks and Learning Systems, 2019.






