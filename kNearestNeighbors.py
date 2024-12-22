from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
print(data)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train, y_train)

print(classifier.score(x_test, y_test))