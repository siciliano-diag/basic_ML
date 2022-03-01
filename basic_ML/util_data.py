#IMPORTS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import resample

import tensorflow_datasets as tfds
#import tensorflow_ranking as tfr


'''
def monk_rules_1(data):
  return np.logical_or((data[:,1] == data[:,2]),(data[:,5] == 1))

def monk_rules_2(data):
  return np.sum([data[j]==1 for j in ["a1","a2","a3","a4","a5","a6"]])==2

def monk_rules_3(data):
  return np.logical_or(np.logical_and(data["a5"]==3,data["a4"]==1),np.logical_and(data["a5"]!=4,data["a2"]!=3))
'''

def bisettrice(x):
  return 1*(x[:,1]>x[:,0])

def quadranti(x):
  return 1*np.logical_or(np.logical_and(x[:,0]>0,x[:,1]>0),np.logical_and(x[:,0]<0,x[:,1]<0))

def parabola(x):
  return 1*(x[:,0]**2-x[:,1]/2-0.25>0)

def circonferenza(x):
  return 1*(x[:,0]**2+x[:,1]**2<0.5)

def cerchi_concentrici(x):
  c1 = x[:,0]**2+x[:,1]**2<0.25**2
  c2 = x[:,0]**2+x[:,1]**2<0.5**2
  c3 = x[:,0]**2+x[:,1]**2<0.75**2
  c4 = x[:,0]**2+x[:,1]**2<1**2
  return 1*np.logical_or(np.logical_or(c1,np.logical_and(c3,np.invert(c2))),np.invert(c4))

def get_custom_data(dataname, selected_seed = 21094):
  samples = 1000
  np.random.seed(selected_seed)
  x = np.random.uniform(-1,1,(samples,2))

  if "bisettrice" in dataname:
    y = bisettrice(x)
  elif "quadranti" in dataname:
    y = quadranti(x)
  elif "parabola" in dataname:
    y = parabola(x)
  elif "circonferenza" in dataname:
    y = circonferenza(x)
  elif "cerchi_concentrici" in dataname:
    y = cerchi_concentrici(x)

  y = y[:,None]

  return x, y

def get_tfds_data(dataname):
  if dataname=="mnist":
    train_data, test_data = tfds.load('mnist',
                                      split=['train', 'test'],
                                      shuffle_files=True, #The MNIST data is only stored in a single file, but for larger datasets with multiple files on disk, it's good practice to shuffle them when training.
                                      as_supervised=True #Returns tuple (img, label) instead of dict {'image': img, 'label': label}
                                      )
    
    train_x = np.stack(list(tfds.as_numpy(train_data.map(lambda x,y: x))))
    train_y = np.stack(list(tfds.as_numpy(train_data.map(lambda x,y: y))))[:,None]
    test_x = np.stack(list(tfds.as_numpy(test_data.map(lambda x,y: x))))
    test_y = np.stack(list(tfds.as_numpy(test_data.map(lambda x,y: y))))[:,None]

  return train_x, test_x, train_y, test_y

def get_local_data(dataname):
  if dataname == "diabetes":
    load_data = pd.read_csv("../data/" + dataname + ".csv")

  x = load_data.iloc[:,:-1].to_numpy()
  y = load_data.iloc[:,-1:].to_numpy()

  return x,y

def download_UCI_united(dataname):
  url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

  if dataname=="german":
    load_data = pd.read_csv(url + "statlog/" + dataname + "/" + dataname +".data-numeric", header=None, sep="\s+")
  elif dataname=="australian":
    load_data = pd.read_csv(url + "statlog/" + dataname + "/" + dataname +".dat", header=None, sep="\s+")
  elif dataname=="covtype":
    load_data = pd.read_csv(url + dataname + "/" + dataname + ".data.gz", header=None)
  elif dataname=="cleveland":
    load_data = pd.read_csv(url + "heart-disease/processed." + dataname + ".data", header=None)
  elif dataname=="crx":
    load_data = pd.read_csv(url + "credit-screening/" + dataname + ".data", header=None)
  elif dataname=="EEG%20Eye%20State":
    load_data = pd.read_csv(url + "00264/" + dataname + ".arff", header=None, skiprows=19)
  elif dataname=="heart":
    load_data = pd.read_csv(url + "statlog/" + dataname + "/" + dataname +".dat", header=None, sep="\s+")
  elif dataname=="sonar":
    load_data = pd.read_csv(url + "undocumented/connectionist-bench/" + dataname + "/" + dataname +".all-data", header=None)
  else:
    load_data = pd.read_csv(url + dataname + "/" + dataname + ".data", header=None)

  if dataname=="car":
    for col in load_data.columns[:-1]:
      load_data[col] = load_data[col].astype("category").cat.codes
  elif dataname=="breast-cancer-wisconsin":
    load_data.drop(0,axis=1,inplace=True)
    load_data.loc[load_data[6]=="?",6] = -1 #replace missing values
    load_data[6] = load_data[6].astype(int)
  elif dataname=="crx":
    app = np.where(load_data=="?")
    for i,j in zip(app[0],app[1]):
      load_data.iloc[i,j] = -1 #replace missing values
    for col in [0,3,4,5,6,8,9,11,12]:
      load_data[col] = load_data[col].astype("category").cat.codes
  elif dataname=="cleveland":
    for col in load_data.columns[:-1]:
      if load_data[col].dtype != float:
        load_data[col] = load_data[col].astype("category").cat.codes
  elif dataname=="glass":
    load_data = load_data.iloc[:,1:] #remove ID column
  elif dataname=="hepatitis":
    app = np.where(load_data=="?")
    for i,j in zip(app[0],app[1]):
      load_data.iloc[i,j] = -1 #replace missing values

  if dataname=="hepatitis": #class is first column
    x = load_data.iloc[:,1:].to_numpy()
    y = load_data.iloc[:,:1].to_numpy()
  else:
    x = load_data.iloc[:,:-1].to_numpy()
    y = load_data.iloc[:,-1:].to_numpy()

  if dataname=="breast-cancer-wisconsin":
    y = (y==4)*1
  elif dataname=="haberman":
    y = (y==2)*1

  return x,y

def download_UCI_divided(dataname):
  url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

  if "monks" in dataname:
    train_data = pd.read_csv(url + "monks-problems/" + dataname + ".train",sep=" ", names = ["c","a1","a2","a3","a4","a5","a6","d"])
    train_data.set_index("d", inplace=True)
    test_data = pd.read_csv(url + "monks-problems/" + dataname + ".test",sep=" ", names = ["c","a1","a2","a3","a4","a5","a6","d"])
    test_data.set_index("d", inplace=True)
  elif dataname == "poker":
    train_data = pd.read_csv(url + dataname + "/poker-hand-training-true.data", header=None)
    test_data = pd.read_csv(url + dataname + "/poker-hand-testing.data", header=None)
    #if redefine_class: #change class to delete noise
    #train_data.iloc[:,0] = monk_rules_3(train_data)*1
  elif dataname == "image":
    train_data = pd.read_csv(url + dataname + "/segmentation"+".data", skiprows=2)
    test_data = pd.read_csv(url + dataname + "/segmentation"+".test", skiprows=2)
  elif dataname=="bupa":
    load_data = pd.read_csv(url + "liver-disorders" + "/" + dataname + ".data", header=None)
    train_data = load_data.loc[load_data.iloc[:,-1]==1,:].iloc[:,:-1]
    test_data = load_data.loc[load_data.iloc[:,-1]==2,:].iloc[:,:-1]
  elif dataname=="adult":
    train_data = pd.read_csv(url + dataname + "/" + dataname + ".data", header=None)
    for col in train_data.columns[:-1]:
      if train_data[col].dtype != int:
        train_data[col] = train_data[col].astype("category").cat.codes
    test_data = pd.read_csv(url + dataname + "/" + dataname + ".test", header=None, skiprows=1)
    test_data.iloc[:,-1] = test_data.iloc[:,-1].apply(lambda x: x[:-1]) #delete last dot char
    for col in test_data.columns[:-1]:
      if test_data[col].dtype != int:
        test_data[col] = test_data[col].astype("category").cat.codes
  
  #divide x, y
  if "monks" in dataname:  #class is first column
    train_x = train_data.iloc[:,1:].to_numpy()
    train_y = train_data.iloc[:,:1].to_numpy()
    test_x = test_data.iloc[:,1:].to_numpy()
    test_y = test_data.iloc[:,:1].to_numpy()
  elif dataname == "image": #class is index
    train_x = train_data.to_numpy()
    train_y = train_data.index.to_numpy()[:,None]
    test_x = test_data.to_numpy()
    test_y = test_data.index.to_numpy()[:,None]
  else: #class is last column
    train_x = train_data.iloc[:,:-1].to_numpy()
    train_y = train_data.iloc[:,-1:].to_numpy()
    test_x = test_data.iloc[:,:-1].to_numpy()
    test_y = test_data.iloc[:,-1:].to_numpy()

  return train_x, test_x, train_y, test_y

def check_y_shape(train_y,test_y = None):
  if (train_y.shape[1]!=1) or (len(np.unique(train_y))>2) or (train_y.dtype not in ["int","float"]): #if 
    enc = OneHotEncoder(sparse=False)

    train_y = np.array(enc.fit_transform(train_y))
    if test_y is not None:
      test_y = np.array(enc.transform(test_y))

    if train_y.shape[1]==2: #keep shape in (1,3,4,...)
      train_y = train_y[:,1:]
      if test_y is not None:
        test_y = test_y[:,1:]
  elif train_y.shape[1]==1 and (not np.array_equal(np.unique(train_y),[0,1])): #if 
    train_y = 1*(train_y==np.max(train_y))
    if test_y is not None:
      test_y = 1*(test_y==np.max(train_y))

  return train_y, test_y

def load_data(dataname = None, test_size = 0.2, re_split_test = False, standardize = True, selected_seed = 21094, **kwargs):
  if dataname is None:
    print("DEFINE DATA NAME")
    return None

  test_x, test_y = None, None

  ###Select data loading
  #already divided in train_test
  if dataname in ["adult","bupa","image","monks-1","monks-2","monks-3","poker"] or dataname in ["mnist"]:
    divided = True

    if dataname in ["mnist"]: #TFDS
      dataloader = get_tfds_data
    else: #UCI
      dataloader = download_UCI_divided

    train_x,test_x,train_y,test_y = dataloader(dataname)
    
  #not divided in train/test
  elif dataname in ["australian","breast-cancer-wisconsin","car","cleveland",
                    "covtype","crx","EEG%20Eye%20State","german","glass",
                    "haberman","heart","hepatitis","ionosphere","iris","sonar"] or dataname in ["diabetes"] or "custom_ours" in dataname:
    divided = False

    if "custom_ours" in dataname: #Custom datasets
      dataloader = lambda x, y=selected_seed: get_custom_data(x, y)
    elif dataname in ["diabetes"]: #local file
      dataloader = get_local_data
    else:
      dataloader = download_UCI_united
    
    train_x,train_y = dataloader(dataname)
  
  else:
    raise NotImplementedError("DATA IMPORT NOT IMPLEMENTED FOR", dataname)

  train_y, test_y = check_y_shape(train_y, test_y)

  if divided and re_split_test:
    train_x = np.concatenate([train_x,test_x], axis=0)
    train_y = np.concatenate([train_y,test_y], axis=0)
    divided = False

  #divide train/test if not already divided and test_size>0
  if test_size>0 and not divided:
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size = test_size, random_state = selected_seed)

  if standardize:
    x_scaler = StandardScaler()
    train_x = x_scaler.fit_transform(train_x)
    if test_x is not None:
      test_x = x_scaler.transform(test_x)
  else:
    x_scaler = None

  min_max_per_feat = np.array([np.min(train_x,axis=0),np.max(train_x,axis=0)])

  return train_x, train_y, test_x, test_y, x_scaler, min_max_per_feat

def bootstrap_accuracy(y,pred_y,n_iters=100):
  app = []
  for i in range(n_iters):
    sub_y, sub_pred_y = resample(y,pred_y)
    app.append(accuracy_score(sub_y,sub_pred_y))
    #print(app[-1])
  return (np.mean(app),np.std(app))

def bootstrap_mse(y,pred_y,n_iters=100):
  app = []
  for i in range(n_iters):
    sub_y, sub_pred_y = resample(y,pred_y)
    app.append(mean_squared_error(sub_y,sub_pred_y))
    #print(app[-1])
  return (np.mean(app),np.std(app))

'''
def bootstrap_ndcg(y,pred_y,n_iters=100):
  app = []
  for i in range(n_iters):
    sub_y, sub_pred_y = resample(y,pred_y)
    app.append(custom_ndcg(sub_y,sub_pred_y))
    #print(app[-1])
  return (np.mean(app),np.std(app))
'''


