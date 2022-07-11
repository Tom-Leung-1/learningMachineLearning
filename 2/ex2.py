# CART on the Bank Note dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#######################################################################################
# Calculate the Gini index for spliting a dataset


def gini_index(left, right, classes):

    giniL = giniR = 1

    if len(left) == 0:
        giniL = 0
    else:
        ####################################################################
        # YOUR CODE HERE!
        # Calculate the Left Subset's Gini Index
        ####################################################################
        sum = 0
        for c in classes:
            p = len(np.where(left[:, -1] == c)[0]) / len(left)
            p = p**2
            sum += p
        giniL = 1 - sum
    if len(right) == 0:
        giniR = 0
    else:
        sum = 0
        for c in classes:
            p = len(np.where(right[:, -1] == c)[0]) / len(right)
            p = p ** 2
            sum += p
        giniR = 1 - sum
        ####################################################################
        # YOUR CODE HERE!
        # Calculate the Right Subset's Gini Index
        ####################################################################

        ####################################################################
        # YOUR CODE HERE!
        # Calculate the Gini Index of the split
        ####################################################################
    total = (len(left)) + (len(right))
    left_no = len(left)
    right_no = len(right)
    gini = giniL * left_no/total + giniR * right_no/total
    return gini


#######################################################################################
# Select the best split point for a dataset
# find the best split point to minmize the gini_index
def split(dataset):

    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_left, b_right = 999, 999, 999, None, None

    # for each feature:
    for index in range(len(dataset[0])-1):
        # test each data' feature as the plane
        for plane in dataset:

            left, right = list(), list()

            ###########################################################################
            # YOUR CODE HERE!
            # Split the dataset by plane[index]'s value to two sets: 'left', 'right'.
            ###########################################################################
            value = plane[index]
            left = dataset[np.where((dataset[:, index] < value) == True)[0],:]
            right = dataset[np.where((dataset[:, index] >= value) == True)[0],:]
            gini = gini_index(left, right, class_values)

            if gini < b_score:
                b_index, b_value, b_score, b_left, b_right = index, plane[index], gini, left, right

    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}


#######################################################################################
# Create a terminal node labelled by majority class in group
# return the majority label
def leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

#######################################################################################
# Build a decision tree


def build_tree(node, max_depth, min_size, depth):

    left, right = node['left'], node['right']

    # check for a no split
    # if not left or not right: # modified
    if not len(left) or not len(right):
        # node['left'] = node['right'] = leaf(left + right) # modified
        node['left'] = node['right'] = leaf(np.concatenate((left, right)))
        return
    # stop split at the max depth
    if depth >= max_depth:
        ###############################################################################
        # YOUR CODE HERE!
        # set both left set and right set as leaves.
        ###############################################################################
        node['left'] = leaf(left)
        node['right']= leaf(right)
        return

    ###################################################################################
    # YOUR CODE HERE!
    # 1. If left group is no more than min_size, set it to a leaf.
    # 2. Else, recur split in the left group.
    ###################################################################################
    if len(left) <= min_size:
        node['left'] = leaf(left)
    else:
        node['left'] = split(left)
        build_tree(node['left'], max_depth, min_size, depth+1)

    # process right child
    if len(right) <= min_size:
        node['right'] = leaf(right)
    else:
        node['right'] = split(right)
        build_tree(node['right'], max_depth, min_size, depth+1)


#######################################################################################
# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

#######################################################################################
# Print the whole decision tree with plain text


def print_decision_tree(node, level=0):
    print(level*'\t'+'Level ', level)
    print(level*'\t'+'Split Feature: ',
          node['index'], ', Split Value: ', node['value'])
    if isinstance(node['left'], dict):
        print(level*'\t'+'In its left part: ')
        print_decision_tree(node['left'], level+1)
    else:
        print(level*'\t'+'In its left part, all nodes are labeled as : ',
              node['left'])
        return
    if isinstance(node['right'], dict):
        print(level*'\t'+'In its right part: ')
        print_decision_tree(node['right'], level+1)
    else:
        print(level*'\t'+'In its right part, all nodes are labeled as : ',
              node['right'])
        return


n_samples = 1000

centers = [(-1, -1), (5, 10), (10, 5)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

######################################################################################
# Append y label to each x sample
train = np.insert(X_train, 2, values=y_train, axis=1)
test = np.insert(X_test, 2, values=y_test, axis=1)


######################################################################################
# Depth and Size setting.

max_depth = 3
min_size = 10


######################################################################################
# Build the decision tree
tree = split(train)
build_tree(tree, max_depth, min_size, 1)


######################################################################################
# Print the tree
print_decision_tree(tree)

######################################################################################
# Predict in the test set
y_predictions = list()
for row in test:
    prediction = predict(tree, row)
    y_predictions.append(prediction)

######################################################################################
# Count the wrong predition
wrong = np.count_nonzero(y_test - y_predictions)
print('Number of wrong predictions is: ' + str(wrong))


######################################################################################
# Plot the decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = list()
for row in grid:
    prediction = predict(tree, row)
    Z.append(prediction)

Z = np.array(Z).reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.get_cmap("RdYlBu"))

######################################################################################
# Plot the training points
for i, color in zip(range(3), "ryb"):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=i,
                cmap=plt.get_cmap("RdYlBu"), edgecolor='black', s=15)

plt.suptitle("The decision tree's surface")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()
