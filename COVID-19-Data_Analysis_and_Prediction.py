#!/usr/bin/env python
# coding: utf-8

# 1.Choisissez une API COVID-19 de votre choix qui contient à la fois des données précieuses et du bruit.

# 2.Utilisez Python pour collecter les données de l'API et les stocker dans un Pandas DataFrame.

# In[1]:


import requests
import pandas as pd


# In[2]:


#API Covid-19 de Johns Hopkins University
url = 'https://api.covid19api.com/summary'
response = requests.get(url)


# In[3]:


data = response.json()
data


# In[4]:


df = pd.DataFrame(data['Countries'], columns = ['Country','TotalConfirmed','NewDeaths','TotalDeaths','NewRecovered','NewConfirmed'])
print(df)


# 3.Nettoyez les données en supprimant les colonnes non pertinentes, les valeurs nulles ou les doublons.

# In[5]:


df.isnull().sum()


# In[6]:


df.drop(['Country'],axis = 1, inplace = True)


# In[7]:


df


# 4.Prétraitez les données en normalisant et en mettant à l'échelle les données numériques.

# In[8]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Normaliser les données numériques
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df)

# Mettre à l'échelle les données numériques
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# In[9]:


scaled_data


# In[10]:


scaled_data[:,0:4]


# In[11]:


scaled_data[:,-1]


# 5.Effectuez une EDA pour identifier les tendances, les corrélations et les modèles dans les données. 
# 	Utilisez des visualisations telles que des histogrammes, des nuages ​​de points et des cartes 
# 	thermiques pour vous aider à mieux comprendre les données.

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


# Visualiser la distribution des données avec des histogrammes
df.hist()
plt.show()


# In[53]:


# Visualiser les relations entre 'TotalDeaths' et 'TotalConfirmed' avec des nuages de points
sns.scatterplot(x='TotalConfirmed',y ='TotalDeaths',  data=df)
plt.show()


# In[54]:


# Visualiser les corrélations entre les variables avec des cartes thermiques
sns.heatmap(df.corr(), annot=True)
plt.show()


# 6.Choisissez l'algorithme supervisé le mieux adapté pour prédire le nombre futur de cas. Utilisez des techniques telles que la division train-test, 
# 	la validation croisée et la recherche de grille pour optimiser les performances du modèle.

# In[55]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[56]:


# Diviser les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(scaled_data[:,0].reshape(-1,1), scaled_data[:,2], test_size=0.2, random_state=42)


# In[57]:


# Entraîner un modèle de régression linéaire sur les données d'entraînement
model = LinearRegression()
model.fit(X_train, y_train)


# In[58]:


# Prédire les valeurs pour les données de test
y_pred = model.predict(X_test)


# In[59]:


# Évaluer les performances du modèle
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Coefficient de détermination R² :', r2)
print('Erreur quadratique moyenne (MSE) :', mse)


# In[60]:


from sklearn.model_selection import GridSearchCV

param_grid = {'fit_intercept': [True, False],
              'normalize': [True, False]}

model = LinearRegression()

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Meilleurs paramètres :', best_params)


# In[61]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print('Scores de validation croisée :', scores)


# In[62]:


#On réentraine le model avec les best_params
model = LinearRegression(**best_params)
model.fit(X_train, y_train)


# In[83]:


#Tester le modèle sur l'ensemble de test et évaluez ses performances à l'aide de la fonction r2_score()
from sklearn.metrics import r2_score

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print('Coefficient de détermination R² :', r2)


# 7.Une fois que vous avez choisi le modèle le mieux adapté, déployez-le à l'aide de Streamlit. Créez une interface conviviale qui permet aux utilisateurs de saisir des données et d'afficher les prédictions du modèle.

# In[87]:


import streamlit as st

st.title('COVID-19 Data Analysis and Prediction')

TotalConfirmed = st.slider("'Total Confirmed'", float(scaled_data[:,0].min()), float(scaled_data[:,0].max()), float(scaled_data[:,0].mean()))

prediction = model.predict([[TotalConfirmed]])

st.write('The future number of cases predicted is', prediction)


# In[ ]:




