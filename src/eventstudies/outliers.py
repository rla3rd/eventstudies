import numpy as np
import pandas as pd
# probabilistic models
import pyod.models.ecod
import pyod.models.mad 
import pyod.models.sos 
# proximity models
import pyod.models.knn 
import pyod.models.lof 
# ensemble models
import pyod.models.iforest


class Outliers:
    def __init__(self, model):
        if model.__module__ in (
            'pyod.models.ecod',
            'pyod.models.iforest',
            'pyod.models.mad',
            'pyod.models.sos',
            'pyod.models.knn',
            'pyod.models.lof'
        ):
            self.model = model()

    def Detect(self, Y):
        self.model.fit(Y)
        scores = self.model.decision_function(Y)
        y_pred = self.model.predict(Y)
        return y_pred, scores

def ECOD(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    pred, scores = Outliers(pyod.models.ecod.ECOD).Detect(Y)
    return pred, scores

def MAD(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    pred, scores = Outliers(pyod.models.mad.MAD).Detect(Y)
    return pred, scores

def SOS(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    pred, scores = Outliers(pyod.models.sos.SOS).Detect(Y)
    return pred, scores

def KNN(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    pred, scores = Outliers(pyod.models.knn.KNN).Detect(Y)
    return pred, scores

def LOF(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    if len(Y) < 20:
        zeros = np.zeros(len(Y))
        scores = zeros
        pred = zeros
    else:
        pred, scores = Outliers(pyod.models.lof.LOF).Detect(Y)
    return pred, scores

def IForest(daily_ret):
    Y = np.array(daily_ret).reshape(-1, 1)
    pred, scores = Outliers(pyod.models.iforest.IForest).Detect(Y)
    return pred, scores











    