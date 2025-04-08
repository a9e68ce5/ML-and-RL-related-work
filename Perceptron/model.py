''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from typing import Protocol, Tuple

import numpy as np
import matplotlib.pyplot as plt

# set the numpy random seed so our randomness is reproducible
np.random.seed(1)


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the Perceptron classes further down.
class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...


class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        self.majority_class = None


    def get_hyperparams(self) -> dict:
        return {}
    

    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''
    
        # YOUR CODE HERE
        unique, counts = np.unique(y, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        if self.majority_class is None:
            raise ValueError("Model has not been trained yet.")
        return [self.majority_class] * len(x)



class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''     

        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu

        # YOUR CODE HERE
        self.w = np.random.uniform(-0.01, 0.01, num_features)
        self.b = np.random.uniform(-0.01, 0.01)
        self.t = 0  # Time step for learning rate decay


    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}
    
    def _train_decay(self, x: np.ndarray, y: np.ndarray, epochs: int , record_learning_curve=True):
        num_examples = x.shape[0]
        num_updates = 0  # Initialize update counter

        for epoch in range(epochs):
            #indices = np.random.permutation(len(y))
            x, y =  shuffle_data(x,y)
            #x , y = x[indices],y[indices]

            for i in range(num_examples):
                learning_rate = self.lr / (1 + self.t) if self.decay_lr else self.lr
                self.t += 0.001
                
                if y[i] * (np.dot(self.w, x[i]) + self.b) < 0:
                    self.w += learning_rate * y[i] * x[i]
                    self.b += learning_rate * y[i]
                    num_updates += 1  # Increment counter
        print(f"Total number of learning rate updates during training: {num_updates}")

    
    def _train_margin(self, x: np.ndarray, y: np.ndarray, epochs: int , record_learning_curve = True):
        
        num_updates = 0
        num_examples = x.shape[0]
        for epoch in range(epochs):
            x, y = shuffle_data(x, y)
            
            for i in range(num_examples):
                learning_rate = self.lr / (1 + self.t) if self.decay_lr else self.lr
                self.t += 0.001
                
                if y[i] * (np.dot(self.w, x[i]) + self.b) < self.mu:
                    self.w += learning_rate * y[i] * x[i]
                    self.b += learning_rate * y[i]
                    num_updates += 1  # Increment counter
        print(f"Total number of learning rate updates during training: {num_updates}")
            
            

            
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''
        #python train.py -m margin --lr 0.1 --mu 10 --epochs 200 
        # evaluate
        # train accuracy: 0.806
        # test accuracy: 0.807
        # YOUR CODE HERE
        num_examples = x.shape[0]
        
        if self.mu == 0:
            self._train_decay(x, y, epochs)
        else:
            self._train_margin(x, y, epochs)
    
    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        return np.sign(np.dot(x, self.w) + self.b)


class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate eta
        '''     

        self.lr = lr
        
        # YOUR CODE HERE
        self.weights = np.random.uniform(-0.01, 0.01, num_features)
        self.bias = np.random.uniform(-0.01, 0.01)
        self.avg_weights = np.zeros(num_features)
        self.avg_bias = 0
        self.count = 1  # Counter to compute the averaged weights

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int , record_learning_curve = True):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        num_examples = x.shape[0]
        num_updates = 0
        for epoch in range(epochs):
            indices = np.random.permutation(num_examples)
            x_shuffled, y_shuffled = x[indices] , y[indices]
            
            for i in range(num_examples):
                prediction = np.dot(self.weights, x_shuffled[i]) + self.bias
                if y_shuffled[i] * prediction <= 0:
                    self.weights += self.lr * y_shuffled[i] * x_shuffled[i]
                    self.bias += self.lr * y_shuffled[i]
                    num_updates += 1
                
                self.avg_weights += self.weights
                self.avg_bias += self.bias
                self.count += 1
        print(f"Total number of learning rate updates during training: {num_updates}")

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        final_weights = self.avg_weights / self.count
        final_bias = self.avg_bias / self.count
        return [1 if np.dot(final_weights, sample) + final_bias > 0 else -1 for sample in x]
    
    
class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        
        # YOUR CODE HERE
        self.weights = np.zeros(num_features)
        self.bias = 0


    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int , record_learning_curve = True):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE 
        num_examples = x.shape[0]
        num_updates = 0
        for epoch in range(epochs):
            indices = np.random.permutation(num_examples)
            x_shuffled, y_shuffled = x[indices], y[indices]
            
            for i in range(num_examples):
                prediction = np.dot(self.weights, x_shuffled[i]) + self.bias
                margin = y_shuffled[i] * prediction
                if margin <= self.mu:
                    eta = (self.mu - margin) / (np.dot(x_shuffled[i], x_shuffled[i]) + 1)
                    self.weights += eta * y_shuffled[i] * x_shuffled[i]
                    self.bias += eta * y_shuffled[i]
                    num_updates += 1
        print(f"Total number of learning rate updates during training: {num_updates}")

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        return [1 if np.dot(self.weights, sample) + self.bias > 0 else -1 for sample in x]

   

# DON'T MODIFY THE FUNCTIONS BELOW!
PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS
def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]
