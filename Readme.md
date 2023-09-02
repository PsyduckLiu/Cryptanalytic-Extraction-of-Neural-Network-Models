# Reproducing the Cryptanalytic Extraction of Neural Network Models

This repository presents an alternative implementation of the model extraction attack discussed in the CRYPTO'20 paper.

The original implementation can be found at the following link:

    https://github.com/google-research/cryptanalytic-model-extraction

The paper can be accessed at the following link:

    https://arxiv.org/abs/2003.04884

## Preparations

To get started you will need to install some dependencies. It should suffice to run

> pip install torch numpy

## Extracting Example Models

First, generate a **ONE-DEEP** model that we will extract by running

> python3 train_model——pytorch.py 10-15-15-1 42

and a trained model will be available in the "models/" directory.

Next, we can employ the **ONE-DEEP** model to generate predictions using input data through the following command:

> python3 trained_model_pytorch_predict.py 10-15-1
