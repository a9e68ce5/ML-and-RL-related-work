''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse

from data import load_data
from evaluate import accuracy
from model import init_perceptron, MajorityBaseline, MODEL_OPTIONS

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    train_x = data_dict['train'].drop('label', axis=1).to_numpy()
    train_y = data_dict['train']['label'].to_numpy()
    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    test_x = data_dict['test'].drop('label', axis=1).to_numpy()
    test_y = data_dict['test']['label'].to_numpy()
    print(f'  test x shape: {test_x.shape}\n  test y shape: {test_y.shape}')

    # load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    if args.model == 'majority_baseline':
        model = MajorityBaseline()

        # train the model
        print(f'train MajorityBaseline')
        model.train(x=train_x, y=train_y)
    
    else:
        model = init_perceptron(
            variant=args.model, 
            num_features=train_x.shape[1], 
            lr=args.lr, 
            mu=args.mu)
        print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

        # train the model
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

    # evaluate model on train and test data
    print('evaluate')
    train_predictions = model.predict(x=train_x)
    train_accuracy = accuracy(labels=train_y, predictions=train_predictions)
    print(f'  train accuracy: {train_accuracy:.3f}')
    test_predictions = model.predict(x=test_x)
    test_accuracy = accuracy(labels=test_y, predictions=test_predictions)
    print(f'  test accuracy: {test_accuracy:.3f}')
