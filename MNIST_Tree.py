from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    X,y = load_digits(return_X_y=True)

    dtc = tree.DecisionTreeClassifier(criterion="entropy")

    dtc.fit(X,y)

    #  Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2, random_state=0)

    y_pred = dtc.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
