from sklearn.base import BaseEstimator

class EmptyTransformer(BaseEstimator):
    def fit(self,X,y):
        return self
    
    def transform(self,X):
        return X
    
    def fit_transform(self,X,y):
        return X
