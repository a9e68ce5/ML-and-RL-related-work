# Decision Tree Project Report

Overview

This document presents the implementation and evaluation of a decision tree classifier using different information gain criteria and depth limits.

Questions and Answers

Q1: Accuracy Calculation (10 points)

Question:
Implement the calculate_accuracy() function to compute model accuracy compared to the ground truth labels.

Answer:
Accuracy is computed as the number of correct predictions divided by the total number of samples. Proper handling of data types (e.g., lists vs. Pandas Series) is crucial to avoid errors like ambiguous truth value checks.

Q2: Majority Baseline Accuracy (10 points)

Question:
Implement the MajorityBaseline model to always predict the majority class. Report train/test accuracy and explain why accuracy might be a bad metric for imbalanced data.

Answer:

Train Accuracy: 0.702

Test Accuracy: 0.699

Issue with Accuracy Metric: In imbalanced datasets, accuracy can be misleading, as predicting the majority class alone may yield high accuracy but fail to identify minority class patterns. Precision, recall, and F1-score provide more comprehensive evaluation.

Q3: Simple Decision Tree (15 points)

Question:
Implement a decision tree without a depth limit, using entropy as the information gain criterion. Report train/test accuracy.

Answer:

Train Accuracy: 1.000

Test Accuracy: 0.855

Analysis: The perfect training accuracy indicates severe overfitting. The model memorized training data but failed to generalize well. Limiting tree depth or applying pruning techniques can enhance generalization.

Q4: Decision Tree with Cross-Validation (15 points)

Question:
Implement a depth limit and use cross-validation with depth values [1, 2, 3, 4, 5, 6] to find the optimal depth. Report the best depth and corresponding accuracy.

Answer:

Optimal Depth: 5

Best Cross-Validation Accuracy: 0.885

Analysis: Depth 5 provides the best balance between bias and variance, preventing overfitting while maintaining a high test accuracy.

Q5: Decision Tree with Best Depth from Cross-Validation (15 points)

Question:
Re-run training and evaluation with the optimal depth limit obtained from cross-validation. Report train and test accuracy.

Answer:

Train Accuracy: 0.905

Test Accuracy: 0.885

Analysis: Limiting the depth to 5 significantly reduces overfitting compared to the fully-grown tree. The test accuracy of 88.5% aligns well with cross-validation results, indicating good generalization.

Q6: Simple Decision Tree w/ Collision Entropy (3 points)

Question:
Implement a decision tree without a depth limit, using Collision Entropy as the information gain criterion. Report train/test accuracy.

Answer:

Train Accuracy: 1.000

Test Accuracy: 0.844

Analysis: The model achieves perfect training accuracy, indicating overfitting, similar to using entropy. The test accuracy is slightly lower than with entropy (0.844 vs. 0.855), suggesting that collision entropy might not capture data patterns as effectively in this case. Further experimentation with pruning and depth control may help mitigate overfitting and improve test performance.

Q7: Decision Tree with Cross-Validation using Collision Entropy (3 points)

Question:
Run cross-validation on the decision tree using collision entropy as the information gain criterion and depth limit values [1, 2, 3, 4, 5, 6]. Report the optimal depth and corresponding accuracy.

Answer:

Optimal Depth: 5

Best Cross-Validation Accuracy: 0.885

Analysis: The cross-validation results for collision entropy are consistent with entropy-based training, indicating that depth 4 is an optimal choice for balancing complexity and generalization.

Q8: Decision Tree with Best Depth from CV using Collision Entropy (4 points)

Question:
Re-run training and evaluation with Collision Entropy and the optimal depth limit obtained from cross-validation. Report train and test accuracy.

Answer:

Train Accuracy: 0.909

Test Accuracy: 0.853

Analysis: The training accuracy slightly increased compared to entropy (0.909 vs. 0.905), suggesting that collision entropy might capture some unique patterns from the data. The test accuracy (0.853) is slightly lower than that with entropy (0.858), meaning collision entropy might generalize slightly worse. Overall, the depth of 5 still helps balance overfitting and generalization.
