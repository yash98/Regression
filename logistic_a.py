import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def g(n):
    return 1/(1+np.exp(-n))


def predict(Xi, Ws):
    Gs = []
    for i in range(len(Ws)):
        WtX = Ws[i].T@Xi
        Gs.append(g(WtX[0][0]))


def gradient_desc(X, Y, Ws, gd_type):
    err2 = 0
    err1 = 0
    Wcp = 0
    while err1 < err2:
        Wc = []
        for i in range(Y.shape[1]):
            Wc.append(np.zeros((1, X.shape[1])))

        for i in range(X.shape[0]):
            Gs = []
            s = 0
            for j in range(len(Ws)):
                WtX = Ws[j].T@X[j:j+1, :]
                sg = g(WtX[0][0])
                s += sg
                Gs.append(sg)
            for j in range(len(Ws)):
                Wc[j] += (Y[i][j]-(Gs[j]/s))*(X[i])

        for i in range(len(Ws)):
            Ws[j] -= eta*Wc[j]


if __name__ == "__main__":
    trainfile = sys.argv[1]
    testfile = sys.argv[2]
    param = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]

    M = np.genfromtxt(trainfile, delimiter=",", dtype=None, encoding=None)
    lel = []
    M1 = 0
    for i in range(M.shape[1]):
        le = pp.LabelEncoder()
        le.fit(M[:, i])
        if i == 0:
            M1 = le.transform(M[:, i])
            M1 = np.reshape(M1, (M.shape[0], 1))
        else:
            tM1 = np.reshape(
                le.transform(M[:, i]), (M.shape[0], 1))
            M1 = np.append(M1, tM1, axis=1)
        lel.append(le)
    enc1 = pp.OneHotEncoder(handle_unknown='ignore')
    enc1.fit(M1[:, :-1])
    X = enc1.transform(M1[:, :-1]).toarray()
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    enc2 = pp.OneHotEncoder(handle_unknown='ignore')
    enc2.fit(M1[:, -1:])
    Y = enc2.transform(M1[:, -1:]).toarray()

    Ws = []
    for i in range(Y.shape[1]):
        Ws.append(np.zeros((1, X.shape[1])))
