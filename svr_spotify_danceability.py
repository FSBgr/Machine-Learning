# **Imports**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

random_state = 1000
df = pd.read_csv("SpotifyDataLarge.csv")

"""# **Preprocessing**"""

df.isna().sum()  # Checking for null values
df = df.dropna()

#df.info() # Checking value types. Everything is int or float.

df.drop('isrc', axis=1, inplace=True)
df.drop('updated_on', axis=1, inplace=True)

"""**Plotting the correlation map**"""

plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")
corr = df.corr()
sns.heatmap(corr,annot=True, cmap="YlGnBu")

"""**Creating and scaling datasets**"""

scale = StandardScaler()
df_sc = scale.fit_transform(df)
df_sc = pd.DataFrame(df_sc, columns=df.columns)

y = df_sc['danceability'] # The aim is to predict the danceability of a song.
X = df_sc.drop('danceability', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.4, random_state=random_state) # Splitting as 60% train test, 40% test set.

"""# **Regression with SVR**"""

SVM_regression = SVR(C=10, kernel='linear')
SVM_regression.fit(X_train, y_train) # training the model.

y_pred = SVM_regression.predict(X_test) # Predicting values

predictions = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred}) # Creating a dataframe with real values and predictions side by side
# predictions.head()

"""**Evaluating the model**"""

MSE_test = round(np.mean(np.square(y_test- y_pred)),2)
RMSE_test = round(np.sqrt(MSE_test),2)
print(RMSE_test, SVM_regression.score(X_test, y_test))

"""Results were: 1000 samples: 0.66
17k 0.64
Therefore with the increasing amount of samples, the mean squared error is decreasing

# **Finding optimal model with Grid Search**
"""

param_grid = {'C': [1,10,100], 'gamma': [1,0.1,0.01], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'], 'degree' : [3,5,9]}

grid = GridSearchCV(estimator=SVR(),param_grid= param_grid, refit = True, verbose=3, cv=2)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_.score(X_train, y_train)

y_pred_best = grid.predict(X_test)

predictions['y_pred_best'] = y_pred_best # Adding best predictions to the dataframe of previous predictions
predictions.head()

"""# **Optimal Model**"""

optimal = SVR(C=1, gamma = 0.1, kernel = 'rbf')

optimal.fit(X_train, y_train)

y_optimal = optimal.predict(X_test)

MSE_opt = round(np.mean(np.square(y_test - y_optimal)),2)
RMSE_opt = round(np.sqrt(MSE_opt),2)
RMSE_opt

plt.plot(X_train['valence'], y_train, 'o')
m, b = np.polyfit(X_train['valence'], y_train, 1)
plt.plot(X_train['valence'], m*X_train['valence']+b)

"""# **Dimensionality Reduction via Feature Selection**

"""

y_pruned = df_sc['danceability']
#df_sc_pruned = df_sc.drop('danceability', axis=1)
#df_sc_pruned  = df_sc_pruned .drop('tempo', axis=1)
#df_sc_pruned  = df_sc_pruned .drop('key', axis=1)
X_pruned = df_sc.drop(['mode', 'key', 'tempo'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_pruned[:100000], y[:100000], test_size=0.4, random_state=random_state)

X_train.shape

optimal.fit(X_train, y_train)

y_optimal = optimal.predict(X_test)

MSE_opt = round(np.mean(np.square(y_test - y_optimal)),2)
RMSE_opt = round(np.sqrt(MSE_opt),2)
RMSE_opt

optimal.score(X_test, y_test)
