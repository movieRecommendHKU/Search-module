import joblib
import numpy as np


class PCAModel:
    def __init__(self):
        self.model = joblib.load('../ModelFiles/pca_model.pkl')
        self.min_val = np.load('../ModelFiles/min_val.npy')
        self.max_val = np.load('../ModelFiles/max_val.npy')


pca = PCAModel()
