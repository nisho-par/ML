from dataset import X_train, y_train, X_test, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)

y_pred = lg_model.predict(X_test)
print('prediction  by logistic regression model: ', y_pred, len(y_pred))
print('real values: ', y_test, len(y_test))

report = classification_report(y_test, y_pred)
print(report)
