import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

data = np.genfromtxt('chips.txt', delimiter=',')
X, y = np.hsplit(data, [-1])
data_train = np.vstack((data[0:47, :], data[58:107, :]))
data_test = np.vstack((data[48:57, :], data[108:117, :]))
X_train, y_train = np.hsplit(data_train, [-1])
X_test, y_test = np.hsplit(data_test, [-1])
(y, y_train, y_test) = (y.ravel(), y_train.ravel(), y_test.ravel())

# A common classifier to which we'll set optimal params
clf = SVC()


def svc_param_selection(X, y):
    param_grid = [
        {
            'kernel': ['linear'],
            'C': [0.001, 0.01, 0.1, 1, 10]
        },
        {
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
    ]
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
    grid_search.fit(X, y)
    return grid_search.best_params_


# Calculate best parameters for SVC using GridSearchCV
best_params = svc_param_selection(X, y)
print(f"best params: {best_params}")

# Fit the model using best params and predict
clf.set_params(**best_params)
clf.fit(X_train, y_train)
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

# Calculate metrics
(precision_train, recall_train, _, _) = \
    precision_recall_fscore_support(y_true=y_train, y_pred=pred_train, average='binary')
(precision_test, recall_test, _, _) = \
    precision_recall_fscore_support(y_true=y_test, y_pred=pred_test, average='binary')
print(f"precision_train:\t{precision_train:5f}, recall_train:\t{recall_train:5f}")
print(f"precision_test:\t\t{precision_test:5f}, recall_test:\t{recall_test:5f}")

# Print results on a plot
# Test data elements will be circled out
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k')

# Draw separating plane on a colored layout
plt.axis('tight')
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)

plt.show()
