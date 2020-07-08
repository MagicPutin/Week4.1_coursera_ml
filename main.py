import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

# pre text processing
X_train = pd.read_csv('Data/_df0abf627c1cd98b7332b285875e7fe9_salary-train.csv')
X_test = pd.read_csv('Data/_d0f655638f1d87a0bdeb3bad26099ecd_salary-test-mini.csv')

y_train = X_train['SalaryNormalized']
X_train = X_train[['FullDescription', 'LocationNormalized', 'ContractTime']]

X_train['FullDescription'] = X_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
X_train['LocationNormalized'] = X_train['LocationNormalized'].fillna('nan')
X_train['ContractTime'] = X_train['ContractTime'].fillna('nan')
X_train['FullDescription'] = X_train['FullDescription'].str.lower()

# vectorizer
vectorizer = TfidfVectorizer(min_df=5)
X_train_vec = vectorizer.fit_transform(X_train['FullDescription'])
X_test_vec = vectorizer.transform(X_test['FullDescription'])

# one-hot
enc = DictVectorizer()
X_train_categ = enc.fit_transform(X_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(X_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# sparse matrix
X_for_train = hstack([X_train_vec, X_train_categ])
X_for_test = hstack([X_test_vec, X_test_categ])

# prediction
ridge = Ridge(alpha=1, random_state=241)
ridge.fit(X_for_train, y_train)
predictions = ridge.predict(X_for_test)

with open('Answers/ans1.txt', 'w') as ans:
    ans.write(str(round(predictions[0], 2)) + ' ' + str(round(predictions[1], 2)))