from dataset import X_train, y_train, X_test, y_test

# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# knn model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# get predictions
y_pred = knn_model.predict(X_test)
print('predictions by knn model: ', y_pred, len(y_pred))
print('real values', y_test, len(y_test))

# create report from real results and predictions
report = classification_report(y_test, y_pred)
print(report)
