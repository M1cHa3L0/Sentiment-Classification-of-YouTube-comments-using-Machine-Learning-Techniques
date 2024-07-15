import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import time


# read file
path = '~/Desktop/Sentiment-Classification-of-YouTube-comments-using-Machine-Learning-Techniques-main/API/filtered_comments_cleaned.txt'

df = pd.read_csv(path, delimiter='\t', low_memory=False)

# remove NA
df = df.dropna(subset=['cleanComment', 'Sentiment'])
print(df.count())

df = df[:3000]

# feature & label
x = df['cleanComment']
y = df['Sentiment']

# TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(x)


# model and hyperparameter
model_params = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [5, 10, 20, 30, 40], # 1-40
            "min_samples_leaf": [1, 2, 5, 10, 15, 20] # 1-20
        }
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 25, 50, 100],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7], # 1-32
        }
    }
}

# hyper tuning and test model
performance_data = []

start_time = time.time()
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp["model"], mp["params"], cv=10, scoring='accuracy', n_jobs=-1)
    clf.fit(tfidf_matrix, y)
    best_model = clf.best_estimator_
    scores = cross_val_score(best_model, tfidf_matrix, y, cv=10, scoring='accuracy')
    mean_score = scores.mean()
    performance_data.append([model_name, *scores, mean_score, clf.best_params_])
    print(f"{model_name} 每折的准确率: {scores}")
    print(f"{model_name} 平均准确率: {mean_score}")
    print(f"{model_name} 最佳超参数: {clf.best_params_}")

end_time = time.time()
print(f"耗时: {end_time - start_time:.4f} 秒")


columns = ['Model'] + [f'Fold_{i+1}' for i in range(10)] + ['Mean Accuracy', 'Best Params']
performance_df = pd.DataFrame(performance_data, columns=columns)
print(performance_df)

# 保存性能数据到CSV文件
performance_df.to_csv('model_performance_with_params.csv', index=False)








