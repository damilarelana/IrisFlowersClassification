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
dataset = pd.read_csv(url, names=names)

# Explore dataset
print(dataset.shape, '\n')  # determine shape of the data
print(dataset.head(20), '\n')  # head to eyeball the data
print(dataset.describe(), '\n')  # descriptions
print(dataset.groupby('class').size(), '\n')  # obtain class distribution

# Visualize Data
sns.set()
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)  # box plot
plt.show()
dataset.hist()  # histogram plot [automatically detects the class/subplots]
plt.show()
scatter_matrix(dataset)  # scatter matrix [using scatter plots] to plot each data frame column against other 3 columns
plt.show()