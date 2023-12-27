
# Preprocessing
# a. Dataset description
# TODO: Provide a brief description of the dataset, including attributes and target attribute.
df = pd.read_csv('penguins.csv')
# b. Remove rows with NaN values
df = df.dropna()

# c. Calculate mean and variance
mean_values = df.mean()
variance_values = df.var()

# Weak estimators
# a. Convert non-numeric discrete attributes
# TODO: Convert non-numeric discrete attributes into numerical values.

# b. Function to get splits
def get_splits(attribute, labels):
    # TODO: Implement the function to identify splits for discretization.
    pass

# c. Number of weak estimators in AdaBoost
# TODO: Determine the number of weak estimators available in AdaBoost.

# AdaBoost
# a. Function to calculate split probabilities
def calculate_split_probs(attribute, split, D):
    # TODO: Implement the function to calculate split probabilities.
    pass

# b. Function to calculate split prediction
def calculate_split_prediction(split_probs):
    # TODO: Implement the function to determine the predicted labels for a split.
    pass

# c. Function to calculate split errors
def calculate_split_errors(split_probs):
    # TODO: Implement the function to calculate the error of a split.
    pass

# d. Function to find the best estimator
def find_best_estimator(df, splits_list, D):
    # TODO: Implement the function to find the attribute and split with the lowest error.
    pass

# e. Attribute, split, and predicted labels with lowest error
# TODO: Determine the attribute, split, and predicted labels with the lowest error.

# f. Function to calculate estimator weight
def get_estimator_weight(estimator_error):
    # TODO: Implement the function to calculate the weight of an estimator.
    pass

# g. Function to update weights
def update_D(df, current_h, D):
    # TODO: Implement the function to update the weights of training instances.
    pass

# h. Function to train AdaBoost model
def adaboost(df, niter):
    # TODO: Implement the function to train an AdaBoost model.
    pass

# i. Function to predict using AdaBoost model
def adaboost_predict(X, model):
    # TODO: Implement the function to predict the label using the AdaBoost model.
    pass

# j. Cross-validation to determine number of iterations
# TODO: Use cross-validation to determine the appropriate number of iterations.

