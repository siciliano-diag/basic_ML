#IMPORTS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import resample

#import tensorflow_datasets as tfds
#import tensorflow_ranking as tfr

def load_data(*args,**kwargs):
	###Select data loading
	if kwargs["from"].lower() == "uci":
		dataloader = download_UCI
	elif kwargs["from"].lower() == "tfds":
		dataloader = get_tfds_data
	elif kwargs["from"].lower() == "local":
		dataloader = get_local_data
	else:
		raise NotImplementedError("DATA IMPORT NOT IMPLEMENTED FOR", dataset)

	data = dataloader(kwargs["name"])

	check_y_shape(data)

	if "x" not in data and kwargs["re_split_test"]:
		for var in ["x","y"]:
			app = []
			for split in ["train","val","test"]:
				key = split+"_"+var
				if key in data:
					app.append(data[key])
					del data[key]
			data[var] = np.concatenate(app, axis=0)
	
	#divide train/test if not already divided and test_size>0
	if "x" in data:
		for var in ["x","y"]:
			data["train_"+var] = data[var]
		
	if kwargs["test_size"]>0 and "text_x" not in data:
		data["train_x"], data["test_x"], data["train_y"], data["test_y"] = train_test_split(data["train_x"], data["train_y"], test_size = kwargs["test_size"], random_state = kwargs["seed"])
		
	if kwargs["val_size"]>0 and "val_x" not in data:
		data["train_x"], data["val_x"], data["train_y"], data["val_y"] = train_test_split(data["train_x"], data["train_y"], test_size = kwargs["val_size"], random_state = kwargs["seed"])

	if kwargs["standardize"]:
		x_scaler = StandardScaler()
		data["train_x"] = x_scaler.fit_transform(data["train_x"])
		for split in ["val","test"]:
			if split in data:
				data[split] = x_scaler.transform(data[split])
	else:
		x_scaler = None

	#min_max_per_feat = np.array([np.min(train_x,axis=0),np.max(train_x,axis=0)])

	return data, x_scaler #min_max_per_feat

def get_local_data(dataset_name):
	if dataset == "diabetes":
		load_data = pd.read_csv("../data/" + dataset_name + ".csv")

	x = load_data.iloc[:,:-1].to_numpy()
	y = load_data.iloc[:,-1:].to_numpy()

	return {"x":x, "y":y}

def download_UCI(dataset_name):
	#divided in train/test
	if dataset_name in ["adult","bupa","image","monks-1","monks-2","monks-3","poker"]:		
		data = download_UCI_divided(dataset_name)
		
	#not divided in train/test
	elif dataset_name in ["australian","breast-cancer-wisconsin","car","cleveland",
										"covtype","crx","EEG%20Eye%20State","german","glass",
										"haberman","heart","hepatitis","ionosphere","iris","sonar"]:
		data = download_UCI_united(dataset_name)
	else:
		print("IMPORT FOR ", dataset_name, "NOT ALREADY DEFINED")

	return data

def download_UCI_united(dataset_name):
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

	if dataset_name=="german":
		load_data = pd.read_csv(url + "statlog/" + dataset_name + "/" + dataset_name +".data-numeric", header=None, sep="\s+")
	elif dataset_name=="australian":
		load_data = pd.read_csv(url + "statlog/" + dataset_name + "/" + dataset_name +".dat", header=None, sep="\s+")
	elif dataset_name=="covtype":
		load_data = pd.read_csv(url + dataset_name + "/" + dataset_name + ".data.gz", header=None)
	elif dataset_name=="cleveland":
		load_data = pd.read_csv(url + "heart-disease/processed." + dataset_name + ".data", header=None)
	elif dataset_name=="crx":
		load_data = pd.read_csv(url + "credit-screening/" + dataset_name + ".data", header=None)
	elif dataset_name=="EEG%20Eye%20State":
		load_data = pd.read_csv(url + "00264/" + dataset_name + ".arff", header=None, skiprows=19)
	elif dataset_name=="heart":
		load_data = pd.read_csv(url + "statlog/" + dataset_name + "/" + dataset_name +".dat", header=None, sep="\s+")
	elif dataset_name=="sonar":
		load_data = pd.read_csv(url + "undocumented/connectionist-bench/" + dataset_name + "/" + dataset_name +".all-data", header=None)
	else:
		load_data = pd.read_csv(url + dataset_name + "/" + dataset_name + ".data", header=None)

	if dataset_name=="car":
		for col in load_data.columns[:-1]:
			load_data[col] = load_data[col].astype("category").cat.codes
	elif dataset_name=="breast-cancer-wisconsin":
		load_data.drop(0,axis=1,inplace=True)
		load_data.loc[load_data[6]=="?",6] = -1 #replace missing values
		load_data[6] = load_data[6].astype(int)
	elif dataset_name=="crx":
		app = np.where(load_data=="?")
		for i,j in zip(app[0],app[1]):
			load_data.iloc[i,j] = -1 #replace missing values
		for col in [0,3,4,5,6,8,9,11,12]:
			load_data[col] = load_data[col].astype("category").cat.codes
	elif dataset_name=="cleveland":
		for col in load_data.columns[:-1]:
			if load_data[col].dtype != float:
				load_data[col] = load_data[col].astype("category").cat.codes
	elif dataset_name=="glass":
		load_data = load_data.iloc[:,1:] #remove ID column
	elif dataset_name=="hepatitis":
		app = np.where(load_data=="?")
		for i,j in zip(app[0],app[1]):
			load_data.iloc[i,j] = -1 #replace missing values

	if dataset_name=="hepatitis": #class is first column
		x = load_data.iloc[:,1:].to_numpy()
		y = load_data.iloc[:,:1].to_numpy()
	else:
		x = load_data.iloc[:,:-1].to_numpy()
		y = load_data.iloc[:,-1:].to_numpy()

	if dataset_name=="breast-cancer-wisconsin":
		y = (y==4)*1
	elif dataset_name=="haberman":
		y = (y==2)*1

	return {"x":x, "y":y}

def download_UCI_divided(dataset_name):
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"

	if "monks" in dataset_name:
		train_data = pd.read_csv(url + "monks-problems/" + dataset_name + ".train",sep=" ", names = ["c","a1","a2","a3","a4","a5","a6","d"])
		train_data.set_index("d", inplace=True)
		test_data = pd.read_csv(url + "monks-problems/" + dataset_name + ".test",sep=" ", names = ["c","a1","a2","a3","a4","a5","a6","d"])
		test_data.set_index("d", inplace=True)
	elif dataset_name == "poker":
		train_data = pd.read_csv(url + dataset_name + "/poker-hand-training-true.data", header=None)
		test_data = pd.read_csv(url + dataset_name + "/poker-hand-testing.data", header=None)
		#if redefine_class: #change class to delete noise
		#train_data.iloc[:,0] = monk_rules_3(train_data)*1
	elif dataset_name == "image":
		train_data = pd.read_csv(url + dataset_name + "/segmentation"+".data", skiprows=2)
		test_data = pd.read_csv(url + dataset_name + "/segmentation"+".test", skiprows=2)
	elif dataset_name=="bupa":
		load_data = pd.read_csv(url + "liver-disorders" + "/" + dataset_name + ".data", header=None)
		train_data = load_data.loc[load_data.iloc[:,-1]==1,:].iloc[:,:-1]
		test_data = load_data.loc[load_data.iloc[:,-1]==2,:].iloc[:,:-1]
	elif dataset_name=="adult":
		train_data = pd.read_csv(url + dataset_name + "/" + dataset_name + ".data", header=None)
		for col in train_data.columns[:-1]:
			if train_data[col].dtype != int:
				train_data[col] = train_data[col].astype("category").cat.codes
		test_data = pd.read_csv(url + dataset_name + "/" + dataset_name + ".test", header=None, skiprows=1)
		test_data.iloc[:,-1] = test_data.iloc[:,-1].apply(lambda x: x[:-1]) #delete last dot char
		for col in test_data.columns[:-1]:
			if test_data[col].dtype != int:
				test_data[col] = test_data[col].astype("category").cat.codes
	
	#divide x, y
	if "monks" in dataset_name:	#class is first column
		train_x = train_data.iloc[:,1:].to_numpy()
		train_y = train_data.iloc[:,:1].to_numpy()
		test_x = test_data.iloc[:,1:].to_numpy()
		test_y = test_data.iloc[:,:1].to_numpy()
	elif dataset_name == "image": #class is index
		train_x = train_data.to_numpy()
		train_y = train_data.index.to_numpy()[:,None]
		test_x = test_data.to_numpy()
		test_y = test_data.index.to_numpy()[:,None]
	else: #class is last column
		train_x = train_data.iloc[:,:-1].to_numpy()
		train_y = train_data.iloc[:,-1:].to_numpy()
		test_x = test_data.iloc[:,:-1].to_numpy()
		test_y = test_data.iloc[:,-1:].to_numpy()

	return {"train_x":train_x, "test_x":test_x, "train_y":train_y, "test_y":test_y}

def check_y_shape(data):
	if "train_y" in data:
		train_y = data["train_y"]
		if "val_y" in data:
			val_y = data["val_y"]
		else:
			val_y = None
		if "test_y" in data:
			test_y = data["test_y"]
		else:
			test_y = None
	elif "y" in data:
		train_y, val_y, test_y = data["y"],None,None

	if (train_y.shape[1]!=1) or (len(np.unique(train_y))>2) or (train_y.dtype not in ["int","float"]): #if 
		enc = OneHotEncoder(sparse=False)

		train_y = np.array(enc.fit_transform(train_y))
		if val_y is not None:
			val_y = np.array(enc.transform(val_y))
		if test_y is not None:
			test_y = np.array(enc.transform(test_y))

		if train_y.shape[1]==2: #keep shape in (1,3,4,...)
			train_y = train_y[:,1:]
			if val_y is not None:
				val_y = val_y[:,1:]
			if test_y is not None:
				test_y = test_y[:,1:]
	elif train_y.shape[1]==1 and (not np.array_equal(np.unique(train_y),[0,1])): #if 
		train_y = 1*(train_y==np.max(train_y))
		if val_y is not None:
			val_y = 1*(val_y==np.max(train_y))
		if test_y is not None:
			test_y = 1*(test_y==np.max(train_y))

	if "train_y" in data:
		data["train_y"] = train_y
		if val_y is not None:
			data["val_y"] = val_y
		if test_y is not None:
			data["test_y"] = test_y
	elif "y" in data:
		data["y"] = train_y


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