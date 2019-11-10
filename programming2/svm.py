import math
import numpy as np

def load_dataset(path):
    with open(path, 'r') as f:
        lines = [l.strip().split(' ') for l in f.readlines()]
    X = np.zeros((len(lines), len(lines[0]) - 1), dtype=float)
    y = np.zeros((len(lines), ), dtype=int)
    for i, l in enumerate(lines):
        y[i] = int(l[0])
        X[i, :] = np.array([float(v.split(':')[1]) for v in l[1:]])
    return X, y


def linear_kernel(X, y):
    pass


def polynominal_kernel(X, y, d, c):
    pass


def rbf_kernel(X, y, gamma):
    pass


if __name__ == '__main__':
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    X, y = load_dataset('vowel.scale')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)

    C = 1.0
    gamma = 1.0
    degree = 3.
    coef0 = 2.4
    eps = 0.01

    def eval_kernel(kernel):
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0)
        model.fit(X_train, y_train)
        X_test_predict = model.predict(X_test)
        acc = (X_test_predict == y_test).sum() / y_test.shape[0]
        return acc

    for k1, k2 in [('linear', linear_kernel),
                   ('poly', lambda x, y: polynominal_kernel(x, y, degree, coef0)),
                   ('rbf', lambda x, y: rbf_kernel(x, y, gamma))]:
        acc1 = eval_kernel(k1)
        acc2 = eval_kernel(k2)

        assert(abs(acc1 - acc2) < eps)
