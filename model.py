import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA

from time import time

def all_same(items):
  return len(set(items)) == 1

# Load training data from csv file
data = pd.read_csv("data/train.csv")

# Extract feature columns
feature_cols = list(data.columns[1:])

# Extract target column 'label'
target_col = data.columns[0]

# Separate the data into feature data and target data (X and y, respectively)
X = data[feature_cols]
y = data[target_col]

# Apply PCA by fitting the data with only 60 dimensions
pca = PCA(n_components=60).fit(X)
# Transform the data using the PCA fit above
X = pca.transform(X)
y = y.values


# Shuffle and split the dataset into the number of training and testing points above
sss = cross_validation.StratifiedShuffleSplit(y, 3, test_size=0.4, random_state=42)
for train_index, test_index in sss:
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

# Fit a KNN classifier on the training set
knn_clf = KNeighborsClassifier(n_neighbors=3, p=2)
knn_clf.fit(X_train, y_train)

# Initialize the array of predicted labels
y_pred = np.empty(len(y_test), dtype=np.int)

start = time()

# Find the nearest neighbors indices for each sample in the test set
kneighbors = knn_clf.kneighbors(X_test, return_distance=False)

# For each set of neighbors indices
for idx, indices in enumerate(kneighbors):
  # Find the actual training samples & their labels
  neighbors = [X_train[i] for i in indices]
  neighbors_labels = [y_train[i] for i in indices]
  
  # if all labels are the same, use it as the prediction
  if all_same(neighbors_labels):
    y_pred[idx] = neighbors_labels[0]
  else:
    # else fit a SVM classifier using the neighbors, and label the test samples
    svm_clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovo', random_state=42)
    svm_clf.fit(neighbors, neighbors_labels)
    label = svm_clf.predict(X_test[idx].reshape(1, -1))

    y_pred[idx] = label
end = time()

print(accuracy_score(y_test, y_pred))
print("Made predictions in {:.4f} seconds.".format(end - start))
