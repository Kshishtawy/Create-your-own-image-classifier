# Build Your Own Image Recognition Tool
## The project is split into two parts
- **First part:** Training an image classifier that can predict different types of flowers then saves the trained model as a checkpoint.
- **Second part:** Creating a command line application that can be used to train new image classifiers or use the same command line applications with checkpoints for predictions

## Prerequisites

The code is implemented in Python 3.6.5, and essential packages include Numpy, Pandas, Matplotlib, PyTorch, PIL, and json. To install PyTorch, visit the official PyTorch website, select your specifications, and follow the provided instructions.

## Command Line Usage

### Training a New Network
- Basic Usage: `python train.py data_directory`
- Displays current epoch, training loss, validation loss, and validation accuracy during the training process.
- Options:
  - Save checkpoints directory: `python train.py data_dir --save_dir save_directory`
  - Choose architecture (alexnet, densenet121, or vgg16 available): `python train.py data_dir --arch "vgg19"`
  - Set hyperparameters: `python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20`
  - Use GPU for training: `python train.py data_dir --gpu gpu`

### Flower Prediction from Image
- Basic usage: `python predict.py /path/to/image checkpoint`
- Options:
  - Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
  - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
  - Use GPU for inference: `python predict.py input checkpoint --gpu`

## GPU Acceleration

Due to the computational intensity of the deep convolutional neural network, training on a standard laptop is impractical. Consider the following options for training:
1. CUDA: Install CUDA if you have an NVIDIA GPU for faster training (time-consuming process).
2. Cloud Services: Explore paid cloud services like AWS or Google Cloud for efficient model training.
3. Google Colab: Utilize Google Colab for free access to a T4 GPU 

After training, the `predict.py` file can efficiently run on a standard CPU, providing rapid results.

## JSON Configuration

To display the flower name, a `.json` file is necessary. This file organizes data into folders with numerical labels corresponding to specific names defined in the `.json` file.

## Hyperparameters

Choosing appropriate hyperparameters can be challenging, given the training time. Consider the following tips:
- Increasing epochs improves training set accuracy but may lead to overfitting.
- Large learning rates yield faster convergence but may overshoot.
- Small learning rates result in higher accuracy but extend learning time.
- Densenet121 is effective for images but requires more training time compared to alexnet or vgg19.

## Pre-Trained Network

The `checkpoint.pth` file contains information about a network trained to recognize 102 different flower species. Specific hyperparameters are crucial for successful predictions. To use the pretrained model for an image located at `/path/to/image`, execute: `python predict.py /path/to/image checkpoint.pth`

## This was the Second project of AI programming with Python Nandegree that I have already graduated from
Confirmation  link: [link](confirm.udacity.com/7JTCKYM9)
![Certificate](https://s3-us-west-2.amazonaws.com/udacity-printer/production/certificates/d371f694-fa60-40a1-9112-a5a9721aa8e7.svg)
