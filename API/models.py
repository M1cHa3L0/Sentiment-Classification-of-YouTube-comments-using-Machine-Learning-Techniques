import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# read file
path = '/Users/apple/Desktop/final project/FinalProject/API/filtered_comments_cleaned.txt'
df = pd.read_csv(path, delimiter='\t', low_memory=False)

# remove NA
df = df.dropna(subset=['cleanComment', 'Sentiment'])
print(df.count())

#df = df[:3000]

# feature & label
x = df['cleanComment']
y = df['Sentiment']

# TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(x)

# decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# 10-fold cross validation
dt_score = cross_val_score(dt_classifier, tfidf_matrix, y, cv=10, scoring='accuracy')

# 10-fold score(accuracy)
print(f"each fold: {dt_score}")
print(f"decision tree mean score: {dt_score.mean()}")

# confusion matrix & report
y_pred = cross_val_predict(dt_classifier, tfidf_matrix, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_mat)
print("\nClassification Report:")
print(classification_report(y, y_pred))






