''' This file defines the model classes that will be used. 
    You need to add your code wherever you see "YOUR CODE HERE".
'''

from math import log2
from typing import Protocol
import numpy as np
import pandas as pd
from collections import Counter


# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the MajorityBaseline and DecisionTree classes further down.
class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...

    def predict(self, x: pd.DataFrame) -> list:
        ...



class MajorityBaseline(Model):
    def __init__(self):
        #pass
        self.most_common_label = None

    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
        '''

        # YOUR CODE HERE
        # print(f'test: {test}')
        #Count occurences of each label in y
        label_counts = Counter(y)

        #Find the most common label
        self.most_common_label = label_counts.most_common(1)[0][0] 
    
    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        
        if self.most_common_label is None :
            raise ValueError ("Model has not been trained yet. Please call 'train before predicting")
        
        #Return the most common label for each row in x
        return [self.most_common_label]*len(x)



class DecisionTree(Model):
    def __init__(self, depth_limit: int = None, ig_criterion: str = 'entropy'):
        '''
        Initialize a new DecisionTree

        Args:
            depth_limit (int): the maximum depth of the learned decision tree. Should be ignored if set to None.
            ig_criterion (str): the information gain criterion to use. Should be one of "entropy" or "collision".
        '''
        
        self.depth_limit = depth_limit
        self.ig_criterion = ig_criterion
        self.tree = None
        print(f"Using information gain criterion: {self.ig_criterion}")

    def _entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -sum((count / total) * log2(count / total) for count in counts.values())

    def _collision_entropy(self, labels):
        total = len(labels)
        counts = Counter(labels)
        return -log2(sum((count / total) ** 2 for count in counts.values()))

    def _information_gain(self, data, target, feature):
        if self.ig_criterion == 'entropy':
            total_entropy = self._entropy(target)
            entropy_func = self._entropy
        else:
            self.ig_criterion == 'collision_entropy'
            total_entropy = self._collision_entropy(target)
            entropy_func = self._collision_entropy
        
        values = data[feature].unique()
        weighted_entropy = sum(
            (len(data[data[feature] == v]) / len(data)) * entropy_func(target[data[feature] == v])
            for v in values
        )
        return total_entropy - weighted_entropy

    def train(self, x: pd.DataFrame, y: list):
        '''
        Train a decision tree from a dataset.

        Args:
            x (pd.DataFrame): a dataframe with the features the tree will be trained from
            y (list): a list with the target labels corresponding to each example

        Note:
            - If you prefer not to use pandas, you can convert a dataframe `df` to a 
              list of dictionaries with `df.to_dict(orient='records')`.
            - Ignore self.depth_limit if it's set to None
            - Use the variable self.ig_criterion to decide whether to calulate information gain 
              with entropy or collision entropy
        '''

        # YOUR CODE HERE
        

        self.tree = self.build_tree(x, pd.Series(y))

    def best_split(self, data, target):
        features = data.columns
        gains = {feature: self._information_gain(data, target, feature) for feature in features}
        return max(gains, key=gains.get)

    def build_tree(self, data, target, depth=0):
        if len(set(target)) == 1:
            return target.iloc[0]
        if len(data.columns) == 0 or (self.depth_limit and depth >= self.depth_limit):
            return target.mode()[0]
        best_feature = self.best_split(data, target)
        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            sub_data = data[data[best_feature] == value].drop(columns=[best_feature])
            sub_target = target[data[best_feature] == value]
            tree[best_feature][value] = self.build_tree(sub_data, sub_target, depth + 1)
        return tree

    def predict_instance(self, instance, tree):
        if not isinstance(tree, dict):
            return tree
        feature = next(iter(tree))
        value = instance.get(feature)
        if value not in tree[feature]:
            return None  # Handle unknown feature values
        return self.predict_instance(instance, tree[feature][value])

    def predict(self, x: pd.DataFrame) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (pd.DataFrame): a dataframe containing the features we want to predict labels for

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE
        return [self.predict_instance(row.to_dict(), self.tree) for _, row in x.iterrows()]
    