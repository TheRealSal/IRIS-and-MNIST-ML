from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    mlp = MLPClassifier(hidden_layer_sizes=(120), max_iter=1000, alpha=1e-4, solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=0.001)

    mlp.fit(X, y)

    y_pred = mlp.predict(X)
    print('Accuracy: %.2f' % accuracy_score(y, y_pred))

    cm = confusion_matrix(y, y_pred)
    print(cm)