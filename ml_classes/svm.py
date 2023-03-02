from dataset import X_train, y_train, X_test, y_test

# support vector machines
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# svm model
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)

# get predictions
y_pred = svm_model.predict(X_test)
print('predictions by svm model: ', y_pred, len(y_pred))
print('real values: ', y_test, len(y_test))

# create report 
report = classification_report(y_test, y_pred)
print(report)
