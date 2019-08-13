import numpy as np
import argparse as ap
from sklearn import linear_model as lm
# from sklearn.preprocessing import PolynomialFeatures
import time

# def a(trainfile, testfile, outputfile, weightfile):
def a(ns):
    M = np.loadtxt(ns.trainfile, delimiter=",")
    X = M[:,:-1]
    # X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = M[:,-1:]
    # TODO: check is saving transpose slower
    Xt = X.T
    W = np.linalg.inv(Xt @ X) @ (Xt @ Y)
    np.savetxt(ns.weightfile, W, delimiter=",")

    X1 = np.loadtxt(ns.testfile, delimiter=",")
    X1 = np.append(np.ones((X1.shape[0], 1)), X1, axis=1)
    Y1 = X1 @ W
    np.savetxt(ns.outputfile, Y1, delimiter=",")

def b(ns):
    M = np.loadtxt(ns.trainfile, delimiter=",")
    X = M[:,:-1]
    # X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = M[:,-1:]

    # load lambdas
    lambdas = np.loadtxt(ns.regularfile)

    # regularization
    min_err = float('Inf')
    min_lambda = 0
    I = np.eye(X.shape[1])
    for i in range(lambdas.shape[0]):
        li = lambdas[i]*I
        # k fold cross validation
        err = 0.0
        for j in range(10):
            # training set
            Xtk = np.append(X[0:int((j/10)*X.shape[0]),:], X[int(((j+1)/10)*X.shape[0]):,:], axis=0)
            Ytk = np.append(Y[0:int((j/10)*Y.shape[0]),:], Y[int(((j+1)/10)*Y.shape[0]):,:], axis=0)

            # validation set
            Xvk = X[int((j/10)*X.shape[0]):int(((j+1)/10)*X.shape[0])]
            Yvk = Y[int((j/10)*Y.shape[0]):int(((j+1)/10)*Y.shape[0])]

            XtkT = Xtk.T
            W = np.linalg.inv((XtkT @ Xtk) + li) @ (XtkT @ Ytk)

            # diff = np.subtract(Yvk, np.matmul(Xvk, W))
            # err += (1/Xvk.shape[0])*np.matmul(np.transpose(diff), diff)

            err += (1/(2*Xvk.shape[0]))*np.linalg.norm(Yvk - (Xvk @ W))
        
        err = err/10
        # print("lambda: ", lambdas[i], "error: ", err)
        if (err<=min_err):
            min_err = err
            min_lambda = lambdas[i]
    
    # this print required for submission
    print(min_lambda)

    # finally W from best lambda on whole training set
    li = min_lambda*I
    Xt = X.T
    W = np.linalg.inv((Xt @ X) + li) @ (Xt @ Y)
    np.savetxt(ns.weightfile, W, delimiter=",")
        
    X1 = np.loadtxt(ns.testfile, delimiter=",")
    X1 = np.append(np.ones((X1.shape[0], 1)), X1, axis=1)
    Y1 = X1 @ W
    np.savetxt(ns.outputfile, Y1, delimiter=",")

def c(ns):
    M = np.loadtxt(ns.trainfile, delimiter=",")
    X = M[:,:-1]
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = M[:,-1:]
    # st = time.time()
    # poly = PolynomialFeatures(2)
    # X = poly.fit_transform(X)
    # print("PF time: ", time.time()-st)
    # print("X ", X.shape)
    lambdas = [1.0e-03, 3.0e-03, 1.0e-02, 3.0e-02, 1.0e-01, 3.0e-01, 1.0e+00, 3.0e+00, 1.0e+01,3.0e+01, 1.0e+02, 3.0e+02, 1.0e+03]
    # lambdas = [1.0e-03]

    min_err = float('Inf')
    min_lambda = 0
    for i in range(len(lambdas)):
        reg = lm.LassoLars(alpha=lambdas[i])
        # st = time.time()
        # k fold cross validation
        err = 0.0
        for j in range(10):
            # print("starting i: ", i, "j: ", j)
            # training set
            Xtk = np.append(X[0:int((j/10)*X.shape[0]),:], X[int(((j+1)/10)*X.shape[0]):,:], axis=0)
            Ytk = np.append(Y[0:int((j/10)*Y.shape[0]),:], Y[int(((j+1)/10)*Y.shape[0]):,:], axis=0)

            # validation set
            Xvk = X[int((j/10)*X.shape[0]):int(((j+1)/10)*X.shape[0])]
            Yvk = Y[int((j/10)*Y.shape[0]):int(((j+1)/10)*Y.shape[0])]

            reg.fit(Xtk, Ytk.ravel())
            W = reg.coef_
            W = np.reshape(W, (Xvk.shape[1], 1))

            diff = Yvk - (Xvk @ W)
            err += (1/(2*Xvk.shape[0]))*(diff.T @ diff)
            
            # err += (1/(2*Xvk.shape[0]))*np.linalg.norm(Yvk - (Xvk @ W))

        err = err/10
        # print("lambda: ", lambdas[i], "error: ", err)
        if (err<=min_err):
            min_err = err
            min_lambda = lambdas[i]
        # print("lambda selection: ", time.time()-st)

    # print(min_lambda)
    
    reg = lm.LassoLars(alpha=min_lambda)
    reg.fit(X, Y.ravel())
    W = reg.coef_
    # np.savetxt(ns.weightfile, W, delimiter=",")
        
    X1 = np.loadtxt(ns.testfile, delimiter=",")
    X1 = np.append(np.ones((X1.shape[0], 1)), X1, axis=1)
    Y1 = X1 @ W
    np.savetxt(ns.outputfile, Y1, delimiter=",")


if __name__ == "__main__":
    p = ap.ArgumentParser(description='Linear Regression')
    sp = p.add_subparsers()

    p_a = sp.add_parser('a')
    p_a.add_argument('trainfile', type=str)
    p_a.add_argument('testfile', type=str)
    p_a.add_argument('outputfile', type=str)
    p_a.add_argument('weightfile', type=str)
    p_a.set_defaults(func=a)

    # b trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt
    # python linear.py b train.csv test_X.csv sample_regularization.txt outputfile weightfile
    p_b = sp.add_parser('b')
    p_b.add_argument('trainfile', type=str)
    p_b.add_argument('testfile', type=str)
    p_b.add_argument('regularfile', type=str)
    p_b.add_argument('outputfile', type=str)
    p_b.add_argument('weightfile', type=str)
    p_b.set_defaults(func=b)

    # python linear.py c train.csv test_X.csv outputfile
    # python linear.py c trainfile.csv testfile.csv outputfile.txt
    p_c = sp.add_parser('c')
    p_c.add_argument('trainfile', type=str)
    p_c.add_argument('testfile', type=str)
    p_c.add_argument('outputfile', type=str)
    p_c.set_defaults(func=c)

    args = p.parse_args()
    args.func(args)
