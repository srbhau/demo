import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image

dataset = pd.read_csv('/content/sample_data/data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,5].values

from sklearn import preprocessing
#labelencoder_X = LabelEncoder()
X = X.apply(preprocessing.LabelEncoder().fit_transform)
print(X)

regressor = DecisionTreeClassifier()
regressor.fit(X.iloc[:,1:5] , y)

X_in = np.array([1,1,0,0])
y_pred = regressor.predict([X_in])
print("Prediction:", y_pred)

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file= dot_data, filled = True, rounded = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
