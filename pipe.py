from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np # linear algebra
import pandas as pd # data processing
import os
import re
import nltk
train = pd.read_csv('./fake-news/train.csv')
test = pd.read_csv('./fake-news/test.csv')
test = test.fillna(' ')
train = train.fillna(' ')
test['total'] = test['title'] + ' ' + test['author'] + test['text']
train['total'] = train['title'] + ' ' + train['author'] + train['text']
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
for index, row in train.iterrows():
 filter_sentence = ''
 sentence = row['total']
 sentence = re.sub(r'[^\w\s]', '', sentence) # cleaning
 words = nltk.word_tokenize(sentence) # tokenization
 words = [w for w in words if not w in stop_words] # stopwords removal
 for word in words:
 filter_sentence = filter_sentence + ' ' + 
str(lemmatizer.lemmatize(word)).lower()
 train.loc[index, 'total'] = filter_sentence
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
train = train[['total', 'label']]
X_train = train['total']
Y_train = train['label']
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
test_counts = count_vectorizer.transform(test['total'].values)
test_tfidf = tfidf.transform(test_counts)
# split in samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, 
random_state=0)
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
print('Accuracy of Lasso classifier on training set: {:.2f}'
 .format(logreg.score(X_train, y_train)))
print('Accuracy of Lasso classifier on test set: {:.2f}'
 .format(logreg.score(X_test, y_test)))
from sklearn.naive_bayes import MultinomialNB
cm = confusion_matrix(y_test, pred)
cm
X_train = train['total']
Y_train = train['label']
pipeline = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer(norm='l2')),
 ('clf', linear_model.LogisticRegression(C=1e5)), ])
pipeline.fit(X_train, Y_train)
filename = 'pipeline.sav'
joblib.dump(pipeline, filename)
filename = './pipeline.sav'
loaded_model = joblib.load(filename)
result = loaded_model.predict
(["flynn hillary clinton big woman campus breitbart daniel j flynnever get feeling life circle roundabout rather head straight line toward intended destination hillary clinton remains big woman campus leafy liberal wellesley massachusetts everywhere else vote likely inauguration dress remainder day way miss havisham forever wore wedding dress speaking great expectations hillary rodham overflowed 48 year ago first addressed wellesley graduating class the president college informed gathered 1969 student needed debate far i could ascertain spokesman kind like democratic primary 2016 minus term unknown even seven sisters school i glad miss adams made clear i speaking today u 400 u miss rodham told classmate after appointing edger bergen charlie mccarthys mortimer snerds attendance bespectacled granny glass awarding matronly wisdom least john lennon wisdom took issue previous speaker despite becoming first win election seat u s senate since reconstruction edward brooke came criticism calling empathy goal protestors criticized tactic though clinton senior thesis saul alinsky lamented black power demagogue elitist arrogance repressive intolerance within new left similar word coming republican necessitated brief rebuttal trust rodham ironically observed 1969 one word i asked class rehearsal wanted say everyone came said talk trust talk lack trust u way feel others talk trust bust what say what say feeling permeates generation perhaps even understood distrusted the trust bust certainly busted clintons 2016 plan "])
print(result)