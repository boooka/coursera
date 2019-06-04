import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('./data/titanic.csv', index_col='PassengerId')
sex_df = pd.DataFrame({
    'Sex': ['male', 'female'],
    'Sexi': [1, 2]})
# df.Sex = df.Sex.replace('^fe(.*)', '0', regex=True).replace('^male', '1', regex=True).astype(int)
df = pd.merge(df, sex_df, how='left', on='Sex')

# df = df[['Pclass', 'Fare', 'Age', 'Sexi', 'Survived']]

df = df.dropna()
futures = ['Sexi', 'Age', 'Fare', 'Pclass', ]
X = df[futures]
y = df['Survived']
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = sorted([
    (v, futures[k]) for k, v in enumerate(clf.feature_importances_)
], reverse=True, key=lambda x: x[0])[:2]
with open('results.txt', 'w') as file:
    file.write(' '.join([i[1] for i in importances]))
