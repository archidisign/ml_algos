# -----------------------------------------------------
# Load Data
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

newsgroups = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.8, test_size=0.2)

# -----------------------------------------------------
# Pre-Processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

vectorizer = CountVectorizer()
vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

tf_idf_vectorizer = TfidfVectorizer()
vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
vectors_test_idf = vectorizer.transform(X_test)

normalizer_train = Normalizer().fit(X=vectors_train)
normalizer_test = Normalizer().fit(X=vectors_test)
vectors_train_normalized = normalizer_train.transform(vectors_train)
vectors_test_normalized = normalizer_train.transform(vectors_test)

# ------------------------------------------------------
# Training
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

clf = MultinomialNB(alpha=.01)
clf.fit(vectors_train, y_train)

# Using Cross-Validation
scores = cross_val_score(clf, vectors_train, y_train, cv=5)
tuned_parameters = [{'alpha': [1, 0.5, 0.2, 0.1]}]
clf = MultinomialNB()
clf = GridSearchCV(clf, tuned_parameters, cv=5, refit=False)
clf.fit(vectors_train, y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
print('scores:',scores)
print('scores_std',scores_std)

# -------------------------------------------------------
# Predict
y_pred = clf.predict(vectors_test)

# -------------------------------------------------------
# Comparison
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
metrics.precision_score(y_test, y_pred, average='macro')
metrics.f1_score(y_test, y_pred, average='macro')
metrics.recall_score(y_test, y_pred, average='macro')
metrics.classification_report(y_test, y_pred)