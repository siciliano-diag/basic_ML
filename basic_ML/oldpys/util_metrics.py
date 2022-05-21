#IMPORTS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import resample

from tensorflow.keras.models import Model
from numpy.linalg import matrix_rank

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC, LinearSVC

def compute_metric(metric, model, train_x, train_y, train_y_pred, test_x = None, test_y = None, test_y_pred = None):
  result = []
  if metric == "accuracy":
    if train_y.shape[-1]==1:
      train_y = train_y[:,0]
      train_y_pred = (train_y_pred[:,0]>0.5)*1

      test_y = test_y[:,0]
      test_y_pred = (test_y_pred[:,0]>0.5)*1
    else:
      train_y = np.argmax(train_y,axis=1)
      train_y_pred = np.argmax(train_y_pred,axis=1)

      test_y = np.argmax(test_y,axis=1)
      test_y_pred = np.argmax(test_y_pred,axis=1)

    result.append(accuracy_score(train_y,train_y_pred))
    result.append(accuracy_score(test_y,test_y_pred))

  elif metric in ["matrix_rank","svm_accuracy"]:
    extractor = Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

    if metric == "matrix_rank":
      metric_func = lambda x,*args: matrix_rank(x)
    elif metric == "svm_accuracy":
      svc_models = []
      metric_func = lambda x,y,i: svc_models[i].score(x, y[:,0])

    for i,(x,y) in enumerate([(train_x,train_y),(test_x,test_y)]):
      features = extractor.predict(x)

      res = []
      for j,representation in enumerate(features[1:-1]): #avoid Input and Output Layers
        if metric == "svm_accuracy" and i==0:
          svc_models.append(SVC(kernel="linear", C=10000, max_iter=100000))
          with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            svc_models[-1].fit(representation, train_y[:,0])
        res.append(metric_func(representation,y,j)) #y,j are ignored by matrix_rank
      result.append(res)
    
  else:
    raise NotImplementedError("METRIC COMPUTATION NOT IMPLEMENTED FOR", metric)

  return result

