''' This file contains functions for evaluating model predictions.
    You don't need to change this file.
'''

def accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    assert len(labels) == len(predictions), (
        f'{len(labels)=} and {len(predictions)=} must be the same length.' + 
        '\n\n  Have you implemented model.predict()?\n'
    )
    
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    accuracy = correct / len(labels)

    return accuracy
