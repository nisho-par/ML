from dataset import X_train, y_train, X_test, y_test

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# naive bayes model
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

# get predictions
y_pred = nb_model.predict(X_test)
print('predictions by naive bayes: ', y_pred, len(y_pred))
print('real values: ', y_test, len(y_test))

# create report 
report = classification_report(y_test, y_pred)
print(report)
