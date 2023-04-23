#Done by EzekielChimana
# tomato-disease-detection
This repository contains code for a tomato disease detection model based on Convolutional Neural Networks (CNNs). The model is trained using a dataset of tomato plant images with different types of diseases. The dataset is split into training and testing subsets, and data augmentation techniques such as rescaling, shearing, zooming, and flipping are used to increase the diversity of the training data.

The model architecture consists of two Conv2D layers with ReLU activation and MaxPooling2D layers, followed by a Flatten layer, two Dense layers with ReLU activation, and a final Dense layer with softmax activation to produce classification probabilities for each disease class. The model is compiled using Adam optimizer and categorical cross-entropy loss, and its performance is evaluated on accuracy metrics.

The code includes callbacks to save the best model during training and stop training early if there is no improvement in validation loss for a certain number of epochs. The trained model is saved as an H5 file and can be used for disease detection on new tomato plant images.

The repository also includes a script to visualize the training and validation accuracy and loss using Matplotlib.

To use this code, download or clone the repository and run the Python script. Make sure to have the necessary dependencies installed and the dataset path correctly specified.
To **watch a demo** of this project, check out this **YouTube video** https://youtu.be/l292-TEWBJ0 .

# **Dataset**

The dataset used to train and test the model is located in the train and test folders, respectively. The dataset consists of 10 different categories of images. The train folder contains 8,000 images and the test folder contains 2,000 images.
# Dependencies

- Python 3.x
- Keras
- TensorFlow
- Matplotlib
# Results
The model achieved an accuracy of 95% on the validation set after training for 20 epochs.
