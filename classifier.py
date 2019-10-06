# Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "./data/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)

# Explore dataset
print(data.shape, '\n')  # determine shape of the data
print(data.head(20), '\n')  # head to eyeball the data
print(data.describe(), '\n')  # descriptions
print(data.groupby('class').size(), '\n')  # obtain class distribution

# Visualize data
sns.set()
data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)  # box plot
plt.show()
data.hist()  # histogram plot [automatically detects the class/subplots]
plt.show()
scatter_matrix(data)  # scatter matrix to plot each dataframe column against other 3 columns
plt.show()

# Model dataset
array = data.values  # extract dataframe values
X = array[:, 0:4]  # extract the flower values
Y = array[:, 4]  # extract the flower classes
validation_size = 0.2
seed = 7
X_training_set, X_validation_set, Y_training_set, Y_validation_set = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness factors
seed = 7
scoring = 'accuracy'  # criteria used to validate the algorithm to be spot-checked

# Test Harness Setup
# to spot the following classification problem algorithms with default settings
# -   Logistic Regression [LR]
# -   Linear Discriminant Analysis [LDA]
# -   K-Nearest Neighbors (KNN)
# -   Classification and Regression Trees [CART]
# -   Gaussian Naive Bayes [NB]
# -   Support Vector Machines [SVM]

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))  # append a tuple of the algorithm and algorithm description
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NG', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
results = []
names = []
for name, model in models:  # iterate through the models so as to run cross-validation for each model i.e. each algorithm
    kfold = model_selection.KFold(n_splits=10, random_state=seed)  # initialize 10-fold cross-validation data splits
    cross_validation_results = model_selection.cross_val_score(model, X_training_set, Y_training_set, cv=kfold, scoring=scoring)
    results.append(cross_validation_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Make Predictions based on selecting KNN as the algorithm
knn = KNeighborsClassifier()
knn.fit(X_training_set, Y_training_set)  # to train
classification_predictions = knn.predict(X_validation_set)  # to test/validate
print(accuracy_score(Y_validation_set, classification_predictions))
print(confusion_matrix(Y_validation_set, classification_predictions))
print(classification_report(Y_validation_set, classification_predictions))
