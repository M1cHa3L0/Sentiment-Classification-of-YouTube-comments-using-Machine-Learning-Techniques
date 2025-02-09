import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import cross_val_score, cross_val_predict




# model and hyperparameter
model_params = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 10, 20, 30], # None
            "min_samples_split": [5, 10, 20, 30, 40], # 20
            "min_samples_leaf": [1, 2, 5, 10, 15, 20] # 1
        }
    },
    "SVM": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10, 100], # 10
            "kernel": ["linear", "rbf", "poly"], # linear
            "gamma": ["scale", "auto"] # scale
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [10, 25, 50, 100], # 50
            "max_depth": [None, 5, 10, 20], # None
            "min_samples_split": [2, 5, 10, 15], # 10
            "min_samples_leaf": [1, 2, 4] # 1
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 250, 300], # 250 [50, 100, 200, 250]
            "learning_rate": [0.1, 0.2], # 0.1
            "max_depth": [5, 10, 15, 20], # 15
        }
    }
    
}



# hyper tuning and test model

def train_model(data):
    tfidf_matrix = data[0]
    y = data[1]
    performance_data = []
    for model_name, mp in model_params.items():
        # initialize grid search
        clf = GridSearchCV(mp["model"], mp["params"], cv=10, scoring='accuracy', n_jobs=-1)
        # train & search
        clf.fit(tfidf_matrix, y)
        # get best model
        best_model = clf.best_estimator_
        scores = cross_val_score(best_model, tfidf_matrix, y, cv=10, scoring='accuracy')
        mean_score = scores.mean()
        performance_data.append([model_name, *scores, mean_score, clf.best_params_])
        print(f"{model_name} each fold accuracy: {scores}")
        print(f"{model_name} average accuracy: {mean_score}")
        print(f"{model_name} best hyperparameter: {clf.best_params_}")
        print('##########')
        
    
    return performance_data





