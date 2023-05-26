from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    X,y = load_iris(return_X_y=True)

    dtc = tree.DecisionTreeClassifier(criterion='entropy')

    dtc.fit(X,y)

    #  Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    y_pred = dtc.predict(X_test)

    print(classification_report(y_test, y_pred))

    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
