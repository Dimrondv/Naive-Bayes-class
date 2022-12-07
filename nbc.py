from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DescretenaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, domain_sizes, laplace=False, logarithms=False, safe_count= False):
        self.domain_sizes_ = domain_sizes
        self.laplace_ = laplace
        self.logarithms_ = logarithms
        self.class_labels_ = None
        self.PY_ = None #Jedno wymiarowy wektor
        self.P_ = None #3 wymiarowy
        self.safe_count_ = safe_count


    def fit(self, X, y):
        self.class_labels_ = np.unique(y)
        m ,n = X.shape
        K = self.class_labels_.size
        yy = np.zeros(m, np.int8)
        for index, label in enumerate(self.class_labels_):
            yy[y == label] = index
        self.PY_ = np.zeros(K)
        for k in range(K):
            #new
            if self.safe_count_:
                self.PY_[k] = np.log10(np.sum(yy==k) / m)
            else:
                self.PY_[k] = np.sum(yy==k) / m
        
        q = np.max(self.domain_sizes_)
        self.P_ = np.zeros((K, n, q))
        for i in range(m):
            for j in range(n):
                self.P_[yy[i], j , X[i,j]] += 1
        
        for k in range(K):
            if self.laplace_:
                for j in range(n):
                    self.P_[k, j] = (self.P_[k, j] + 1) / (self.PY_[k] * m + self.domain_sizes_[j])
            else:
                self.P_[k] /= self.PY_[k] * m

    def predict(self, X):
        return self.class_labels_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        m, n = X.shape
        K = self.class_labels_.size
        scores = np.zeros((m, K))
        for i in range(m):
            for k in range(K):
                scores[i, k] =self.PY_[k]
                for j in range(n):
                    # new
                    if self.safe_count_:
                        scores[i, k] *= self.P_[k, j, X[i, j]]
                    else:
                        scores[i, k] += self.P_[k, j, X[i, j]]
                scores[i, k] *= self.PY_[k]
            s = scores[i].sum()
            if s > 0:
                scores[i] /= s
        return scores