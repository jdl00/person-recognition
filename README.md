# Person indentifier

This project uses a Convolutional neural network to perform multiple task classifications. Each of the tasks has a fully connected neural network connected to a shared fully connected neural network.

## Contents
1. [About](#about)
2. [Setup](#setup)
3. [Usage](#usage)
5. [References](#references)
6. [License](#license)

## About

This project was created as a research project into Multi-label Object Attribute Classification. It features a 4 convolutional blocks connected to a shared fully connected network, with each label containing its own fully connected network.

### Dataset

The model was trained on a 48x48 greyscale images containing ethnicity, gender and age (*[Dataset](kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv)*). For age, to fix missing issues the data is converted into categorical data of roughly equal distributions *see datasets/clean_dataset.csv*

### Model

#### ConvBlock:

This is a building block for handling images. It performs a series of operations commonly used in image processing:

- Convolution: Applies filters to the image to extract features like edges, textures, etc.
- Batch Normalization: Stabilizes learning by normalizing the output of the convolution layers.
- ReLU Activation: Introduces non-linearity, allowing the model to learn more complex patterns.
- Pooling: Reduces the size of the feature maps to decrease computation and control overfitting.
- Dropout: Randomly drops units during training to prevent over-dependence on specific paths, enhancing generalization.

#### SharedFC:

A fully connected layer that processes flattened data (like the output of ConvBlock layers).

- Acts as a shared processor for different tasks, learning features that are useful across all tasks.
- Includes ReLU activations and dropout for non-linearity and regularization.

#### FCOutput:

Another fully connected layer designed to output predictions for specific tasks like age, ethnicity, or gender.

 - Processes the features extracted by shared layers to make final predictions for each specific task.
- Also includes ReLU activations and dropout for effective learning.

#### MultiOutputConv:

This is the main model combining all the above components. It first uses a series of ConvBlocks to process the input image, extracting increasingly complex features at each stage.

- The output of these blocks is then flattened and passed through SharedFC to learn shared representations.
- Finally, it splits into task-specific paths, each passing through an FCOutput to generate predictions for age, ethnicity, and gender.


## Setup

1. Clone the github repo \
`git clone github.com/jdl00/person-recognition`

2. Setup and activate virtual environment \
`python -m venv .venv` \

3. Install the requirements \
`pip install -r requirements.txt`

## Usage

1. Activate the environment  \
`source .venv/bin/activate`

2. Train the model using the dataset above \
`python train.py`

3. Run the inference on the model \
`python main.py `

## References

1. [Multi-label Object Attribute Classification using a Convolutional Neural Network](https://arxiv.org/abs/1811.04309)
2. [Multi-label convolutional neural network based pedestrian attribute classification](https://www.sciencedirect.com/science/article/abs/pii/S0262885616301135)

## License
**see LICENSE.md**