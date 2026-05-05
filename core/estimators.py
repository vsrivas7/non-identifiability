from sklearn.linear_model import LogisticRegression

def train_logreg(X, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf

def predict(clf, X):
    return clf.predict_proba(X)[:,1]