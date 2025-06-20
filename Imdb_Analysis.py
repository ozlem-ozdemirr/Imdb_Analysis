# 1. Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

sns.set_style("whitegrid")
%matplotlib inline

# 2. Data Loading
df = pd.read_csv('imdb_top_1000.csv')
df.head()

# 3. Data Review
print(df.info())
print(df.isnull().sum())
print(df.columns)

# 4. Data Cleansing

# Runtime (exp: "142 min" -> 142)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)

# Main Genre 
df['Main_Genre'] = df['Genre'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x)

# Gross (hasılat)
df['Gross'] = df['Gross'].replace('[\$,]', '', regex=True)
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

# Budget (bütçe)
df['Budget'] = df['Budget'].replace('[\$,]', '', regex=True)
df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')

# Director (yönetmen) - first director
df['Director'] = df['Director'].fillna('Unknown')
df['Main_Director'] = df['Director'].apply(lambda x: x.split(',')[0])

# Actors (oyuncular) - Number of actors
df['Actors'] = df['Actors'].fillna('')
df['Num_Actors'] = df['Actors'].apply(lambda x: len(x.split(',')) if x else 0)

# Released_Year convert to number
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')


# 5. Fundamental Analysis and Visualization

plt.figure(figsize=(10,6))
sns.countplot(data=df, y='Main_Genre', order=df['Main_Genre'].value_counts().index[:10], palette='mako')
plt.title('Most Popular Movie Genres (Top 10)')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
genre_rating = df.groupby('Main_Genre')['IMDB_Rating'].mean().sort_values(ascending=False)
sns.barplot(x=genre_rating.values, y=genre_rating.index, palette='viridis')
plt.title('Genres According to Average IMDB Rating')
plt.xlabel('Average IMDB Rating')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df['Released_Year'].dropna(), bins=30, kde=False, color='coral')
plt.title('Number of Movies by Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='IMDB_Rating', y='Gross', hue='Main_Genre', alpha=0.7)
plt.title('Relationship Between IMDb Rating and Gross')
plt.xlabel('IMDb Puanı')
plt.ylabel('Gross (USD)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
correlation = df[['IMDB_Rating', 'Runtime', 'Gross', 'Budget']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Numerical Variables')
plt.tight_layout()
plt.show()

# 6. Genre-Based IMDb Rating Prediction Model

df_genre = df[['Main_Genre', 'IMDB_Rating']].dropna()
X = df_genre[['Main_Genre']]
y = df_genre['IMDB_Rating']

encoder_genre = OneHotEncoder(sparse=False)
X_encoded = encoder_genre.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model_genre = LinearRegression()
model_genre.fit(X_train, y_train)

y_pred = model_genre.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# 6. Genre-Based IMDb Rating Prediction Model

df_genre = df[['Main_Genre', 'IMDB_Rating']].dropna()
X = df_genre[['Main_Genre']]
y = df_genre['IMDB_Rating']

encoder_genre = OneHotEncoder(sparse=False)
X_encoded = encoder_genre.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model_genre = LinearRegression()
model_genre.fit(X_train, y_train)

y_pred = model_genre.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Genre-Based IMDb Rating Prediction Model RMSE: {rmse:.3f}')
print(f'R2 Score: {r2:.3f}')
print(f'Genre-Based IMDb Rating Prediction Model RMSE: {rmse:.3f}')
print(f'R2 Score: {r2:.3f}')