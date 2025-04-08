# Assignment 2

Functions to implement:

```
model.py
    MajorityBaseline.train()
    MajorityBaseline.predict()
    Perceptron.train()
    Perceptron.predict()
    AveragedPerceptron.train()
    AveragedPerceptron.predict()
    AggressivePerceptron.train()
    AggressivePerceptron.predict()

cross_validation.py
    cross_validation()

epochs.py
    train_epochs()
```


## Models

### Majority Baseline

Once you've implemented `MajorityBaseline`, you can train and evaluate your model with:
```sh
python train.py -m majority_baseline
```
Make sure your code works for `MajorityBaseline` before moving on to `Perceptron`. 

### Perceptron

The `Perceptron` class will handle the Simple Perceptron, the Decaying Learning Rate Perceptron, and the Margin Perceptron. You can initialize these in the `train.py` script with the `-m` flag set to `simple`, `decay`, or `margin`. The `--lr` and `--mu` flags set the learning rate and mu hyperparameters, respectively. For the `simple` and `decay` models, you can either ignore the `self.mu` class variable, or use it in your linear threshold with `mu=0`. For the decay model, the `lr` parameter is the same as the lr0 hyperparameter. The `num_features` parameter will let you determine the number of weights your model will have. It's up to you whether you have an explicit bias term or whether you fold it into your weights.

Once you've implemented `Perceptron`, you can train and evaluate your model like this:
```sh
# train/eval a simple perceptron with lr=1 for 10 epochs
python train.py -m simple --lr 1 --epochs 10

# train/eval a decay perceptron with lr=1 for 10 epochs
python train.py -m decay --lr 1 --epochs 10

# train/eval a margin perceptron with lr=1 and mu=1 for 10 epochs
python train.py -m margin --lr 1 --mu 0.1 --epochs 10
```

### Averaged Perceptron

The `AveragedPerceptron` class will handle the Averaged Perceptron. Once you've implemented `AveragedPerceptron`, you can train and evaluate your model like this:
```sh
# train/eval an averaged perceptron with lr=1 for 10 epochs
python train.py -m averaged --lr 1 --epochs 10
```

### Aggressive Perceptron (CS 6350 only)

The `AggressivePerceptron` class will handle the Averaged Perceptron. You can leave this class as-is if you're not in CS 6350. Once you've implemented `AggressivePerceptron`, you can train and evaluate your model like this:
```sh
# train/eval an averaged perceptron with mu=1 for 10 epochs
python train.py -m aggressive --mu 1 --epochs 10
```

## Cross Validation

This is similar to the previous assignment, except you'll have to perform grid search over multiple combinations of hypterparameters. Once you've implemented the necessary code in `cross_validation.py`, you can run cross validation with:
```sh
# run cross validation for a margin perceptron with lr values [1, 0.1, 0.01]
# and mu values [1, 0.1] for 10 epochs
python cross_validation.py -m margin --lr_values 1 0.1 0.01 --mu_values 1 0.1 --epochs 10
```

## Epoch Training

In this part, you'll use the train and validation datasets to see how long to train your perceptron for. You'll want to train the same model for one epoch at a time, evaluate against the validation dataset, and see which epoch yields the best validation accuracy. Beware of "off-by-one" errors. Once you've implemented the necessary code in `epochs.py`, you can run cross validation with:
```sh
# run epoch training for a margin perceptron with lr=1 and mu=1 for 20 epochs
python epochs.py -m margin --lr 1 --mu 1 --epochs 20
```
