# maybe-neural-network
MaybeNeuralNetwork (MNN) - a Bayesian Convolutional Neural Network for image classification that is designed to estimate uncertainty in predictions. It'll *probably* know what its looking at!

This project includes:
	 * A PyTorch implementation of a Bayesian Neural Network (BNN)
	 * Utilities to visualize prediction confidence and entropy
	 * A series of notebooks that show the development from basic neural nets to full Bayesian models, both with and without PyTorch

Note that while this can be imported as a python module the notebook `torch_mnist_bnn.ipynb` includes explanations for the functions and methods to implement the BNN. Furthermore, as the project development section will go over, the additional two notebooks show the building process to a full BNN. If one would like to see the full mathematical implementations without PyTorch, `numpy_mnist_bnn.ipynb` would be the notebook to look at!

## Methodology
1. Bayesian Linear Layer

In a Bayesian Neural Network (BNN), the weights are treated as random variables instead of fixed parameters. Each weight $w$ is modeled using a probability distribution — typically a normal distribution with learnable mean $\mu$ and variance $\sigma^2$. This setup allows the model to estimate its uncertainty.

2. Probabilistic Inference

At inference time, weights are sampled multiple times from their learned distributions. The network’s predictions are gathered across these samples, which gives a sense of both the confidence and the entropy of its predictions.

## Project Development
This project evolved over several stages:

1. `mnist_nn.ipynb`: A basic neural network using to classify MNIST digits.

2. `numpy_mnist_bnn.ipynb`: A full Bayesian neural network built entirely from scratch using NumPy, with no use of autograd or PyTorch.

3. `torch_mnist_bnn.ipynb`: The final version using PyTorch to implement a scalable Bayesian model, incorporating:
   - Custom Bayesian layers  
   - Uncertainty measures (confidence, entropy)  
   - Visualizations of prediction distribution and error bars


## Create Conda Environment 
```
# to activate 
conda create --name mnn python=3.10 -y
conda activate mnn

# install dependencies
pip install -r requirements.txt

# to deactivate
conda deactivate
```

## Installation 
To install as a python module:
```
git clone https://github.com/yourusername/maybe-neural-network.git
cd maybe-neural-network
pip install -e .
```
## Usage
To run the training and evaluation pipeline 
```
python main.py
```

```
```
