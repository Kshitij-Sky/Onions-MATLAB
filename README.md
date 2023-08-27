# Overview
This MATLAB code is designed to process images of onions and weeds, extract shape and texture features from them, and then use those features to classify the objects as either onions or weeds using Support Vector Machines (SVMs). Below is a detailed overview of the script's functionality.

The code starts by initializing variables and reading image files of RGB, depth, and truth (ground truth) using a loop. The images are processed to compute both shape and texture features. The shape features include circularity, eccentricity, solidity, and extent, while the texture features involve contrast, correlation, energy, and homogeneity, computed using Gray-Level Co-occurrence Matrices (GLCMs).

The extracted features are then plotted for analysis. Shape features are plotted in histograms for circularity, eccentricity, solidity, and extent. Texture features are plotted separately for each color channel (red, green, blue), and for the depth image.

After feature extraction and visualization, the code proceeds to perform classification using Support Vector Machines (SVMs). Features are divided into training and test sets. Three different scenarios are commented and available for use:

1. Shape features only.
2. Texture features only.
3. Both shape and texture features combined.
4. Top 5 features based on highest weights (using Neighborhood Component Analysis)

The code includes the training of an SVM model using the `fitcsvm` function. Feature weights are plotted to determine the most influential features for classification.

Additionally, there's a prediction step using the trained SVM model with test data to classify objects as "onions" or "weeds".

Overall, this code showcases a pipeline for image processing, feature extraction, and classification using SVMs for the purpose of distinguishing between onions and weeds in images. Images 1-18 are used for training and 19,20 are used for testing.

The feature weights based on NCA are as follows:

![image](Results/Feature_Weights.png)
