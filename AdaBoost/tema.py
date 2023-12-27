import pandas as pd
import numpy as np



# Dataset
dataFrame = pd.read_csv('https://raw.githubusercontent.com/Vali2804/ML_TemaPractica/main/penguins.csv?token=GHSAT0AAAAAACIMOBTBW77MYYHTDBHKFRXGZMHA2PQ')

# 1. Preprocessing
# Dataset Description 

#a.
# Penguin Dataset Description
attributes = ['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
targetAttribute = 'species'
purpose = 'Classification of penguin species'

# Attribute types
discreteAttributes = ['species', 'island']
continuousAttributes = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# b.
# Remove NaN values
dataFrame = dataFrame.dropna()

# c.
# Calculte mean and variance for each attribute
num_attributes = dataFrame[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
mean_values = num_attributes.mean()
variance_values = num_attributes.var()

print('Mean values for each attribute:\n', mean_values)
print('Variance values for each attribute:\n', variance_values)

# 2. Weak estimators
# a. convert discrete attributes to numeric
dataFrame['species'] = dataFrame['species'].astype('category').cat.codes
dataFrame['island'] = dataFrame['island'].astype('category').cat.codes

# b. the function get_splits()
def get_splits(attribute_values, labels):
    unique_values = np.unique(attribute_values)
    splits = []

    for value in unique_values:
        left_mask = attribute_values == value
        right_mask = ~left_mask

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]

        split = (left_labels.mean() + right_labels.mean()) / 2
        splits.append((value, split))

    return splits

# c. the number of weak estimator ( splits ) in our verison of AdaBoost is 4

# 3. AdaBoost

# a. the function calculate_split_probs()

def calculate_split_probs(attribute, split, D):
    left_weights = {}
    right_weights = {}

    # Subset the dataframe based on the split condition
    left_subset = dataFrame[dataFrame[attribute] <= split]
    right_subset = dataFrame[dataFrame[attribute] > split]

    # Calculate the sum of weights for each label in the left subset
    for label in left_subset[targetAttribute].unique():
        left_weights[label] = sum(D[left_subset[targetAttribute] == label])

    # Calculate the sum of weights for each label in the right subset
    for label in right_subset[targetAttribute].unique():
        right_weights[label] = sum(D[right_subset[targetAttribute] == label])

    return {
        "left": left_weights,
        "right": right_weights
    }

# b. the function calculate_split_prediction()
def calculate_split_prediction(split_probs):
    left_weights = split_probs["left"]
    right_weights = split_probs["right"]

    label_left = max(left_weights, key=left_weights.get)
    label_right = max(right_weights, key=right_weights.get)

    return [label_left, label_right]

# c. the function calculate_split_error()

def calculate_split_errors(split_probs):
    left_weights = split_probs["left"]
    right_weights = split_probs["right"]

    max_left_weight = max(left_weights.values())
    max_right_weight = max(right_weights.values())

    split_error = 1 - max(max_left_weight, max_right_weight)
    return split_error

# d. the function find_best_estimator()

def find_best_estimator(df, splits_list, D):
    best_attribute = None
    best_split = None
    best_label_left = None
    best_label_right = None
    best_error = float('inf')

    for attribute_index, splits in enumerate(splits_list):
        for split in splits:
            split_probs = calculate_split_probs(df.columns[attribute_index], split, D)
            split_prediction = calculate_split_prediction(split_probs)
            split_error = calculate_split_errors(split_probs)

            if split_error < best_error:
                best_attribute = attribute_index
                best_split = split
                best_label_left, best_label_right = split_prediction
                best_error = split_error

    return [(best_attribute, best_split, best_label_left, best_label_right), best_error]


# e. Assuming that D follows the uniform distribution, what is the attribute and the split that achieves the lowest error? What are the labels predicted by the split?

# f. the function get_estimator_weight()

def get_estimator_weight(error):
    alpha = 0.5 * np.log((1 - error) / error)
    return alpha

# g. the function update_D()

def update_D(df, current_h, D):
    predictions = current_h.predict(df)  # Assuming current_h is a trained estimator with a predict() method
    misclassified = predictions != df[targetAttribute]  # Assuming targetAttribute is defined

    updated_D = D.copy()
    updated_D[misclassified] *= np.exp(current_h.weight)  # Assuming current_h has a weight attribute

    updated_D /= np.sum(updated_D)  # Normalize the weights

    return updated_D


# h. the function AdaBoost()

def adaboost(df, niter):
    # Initialize weights
    D = np.ones(len(df)) / len(df)
    
    estimators = []
    estimators_weights = []
    
    for _ in range(niter):
        # Find the best estimator
        splits_list = get_splits(df[continuousAttributes].values, df['species'].values)
        best_estimator, error = find_best_estimator(df, splits_list, D)
        
        # Calculate estimator weight
        weight = get_estimator_weight(error)
        
        # Update weights
        D = update_D(df, best_estimator, D)
        
        # Save estimator and weight
        estimators.append(best_estimator)
        estimators_weights.append(weight)
    
    return {
        "estimators": estimators,
        "estimators_weights": estimators_weights
    }

# i. the function adaboost_predict()

def adaboost_predict(X, model):
    predictions = []
    
    for estimator, weight in zip(model["estimators"], model["estimators_weights"]):
        prediction = estimator.predict(X)
        predictions.append(prediction)
    
    majority_vote = np.sign(np.sum(predictions, axis=0))
    return majority_vote

# j. Use cross-validation to identify the appropriate number of iterations that will achieve good results on your dataset. The number of iterations shouldn't exceed 50.

def cross_validation(df, n_folds, max_iterations):
    fold_size = len(df) // n_folds
    best_iterations = 0
    best_accuracy = 0
    
    for iterations in range(1, max_iterations + 1):
        accuracies = []
        
        for fold in range(n_folds):
            # Split the data into training and validation sets
            validation_start = fold * fold_size
            validation_end = validation_start + fold_size
            validation_set = df[validation_start:validation_end]
            training_set = pd.concat([df[:validation_start], df[validation_end:]])
            
            # Train the AdaBoost model
            model = adaboost(training_set, iterations)
            
            # Make predictions on the validation set
            predictions = adaboost_predict(validation_set[continuousAttributes].values, model)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == validation_set['species'].values)
            accuracies.append(accuracy)
        
        # Calculate average accuracy across folds
        average_accuracy = np.mean(accuracies)
        
        # Check if this iteration achieves better accuracy
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_iterations = iterations
        
    return best_iterations

# Usage example
n_folds = 5
max_iterations = 50
best_iterations = cross_validation(dataFrame, n_folds, max_iterations)
print("Best number of iterations:", best_iterations)



# k. the function adaboost implemented with sklearn