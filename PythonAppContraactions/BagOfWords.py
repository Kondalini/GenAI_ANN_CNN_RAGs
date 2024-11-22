
from numpy.random import multinomial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('spam.csv')
print(df.head(5))

df.Category.value_counts()
df.Category.value_counts()/len(df)*100
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head(5)

#directlly 

#new_df = df['Category'].replace({'ham':0, 'spam':1},inplace=True)

print(df.shape)

#Train test split 
x_train,x_test, y_train, y_test = train_test_split(df.Message,df.spam, test_size =0.2)

x_train.hsape
x_test.shape

#create a bag of words representation using CountVectorize
v = CountVectorizer
x_train_cv = v.fit_transform(x_train.values)
x_test_cv = v.fit_transform(x_test.values)
x_train_cv.toarray()[:2][0]
x_train.shape


v.get_feature_names_out()[1000]
x_train_np = x_train_cv.toarray()
x_train_np[0]

np.where(x_train_np[0]!=0)

model = MultinomialNB(
    model.fit(x_train_cv, y_train))

y_pred = model.predict(x_test_cv)
print(classification_report(y_test,y_pred))