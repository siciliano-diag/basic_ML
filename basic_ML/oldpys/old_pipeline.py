#IMPORTS
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import json

#import tensorflow as tf
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score, mean_squared_error

from datetime import datetime
#from sklearn.utils.class_weight import compute_class_weight
#from tensorflow.keras.models import load_model

import util_data
from util_metrics import *

from copy import deepcopy


from util_cfg import load_configuration, handle_reference


class Pipeline():
	def __init__(self):
		self.cfg = load_configuration()

		self.set_cfg_to_self()
		for key in ["exp","pipeline"]: #Pipeline and exp are not part of the config
			del self.cfg[key] 

		if self.verbose:
			print(self.cfg)

	def set_cfg_to_self(self):
		for key, value in self.cfg.items():
			setattr(self, key, value)
		
		if not hasattr(self, 'verbose'):
			self.verbose = False

	#def prepare_experiments(self):
		#if "seed" not in self.cfg["exp"]:
		#	self.cfg["exp"]["seed"] = 42

		#if "verbose" not in self.cfg["exp"]:
		#	self.cfg["exp"]["verbose"] = 0

	def run(self, start_cmd = 0, end_cmd = None, pipeline = None):
		pipeline = self.pipeline if pipeline is None else pipeline

		for command in pipeline[start_cmd:end_cmd]:
			print("EXECUTING:", command)
			if isinstance(command,str):
				command_name, args, kwargs, outs = command, [], {}, []
			elif isinstance(command,dict):
				command_name = list(command.keys())[0]
				if  isinstance(command[command_name],dict):
					args = self.get_key_if_exists(command[command_name],"args",[])
					kwargs = self.get_key_if_exists(command[command_name],"kwargs",{})
					outs = self.get_key_if_exists(command[command_name],"outs",[])
				else:
					args, kwargs, outs = [], {}, []
			else:
				print("ERROR: command not str or dict")

			if " " in command_name: #check if special words
				command_split = command_name.split(" ")
				if len(command_split)>2:
					print("!!!ERROR: COMMAND LENGTH>2!!!")
				
				if command_split[0]=="sweep":
					key = command_split[1]
					sweep_params = self.cfg.get_composite_key(key)
					for value in sweep_params:
						self.set_composite_key(key,value)
						self.cfg.set_composite_key(key,value) #change configuration to save/check experiments
						out = self.run(pipeline = command[command_name])
					self.cfg.set_composite_key(key,sweep_params) #revert key change to original sweep params
				elif command_split[0]=="repeat":
					num_repeats = int(command_split[1])
					for repeat_i in range(num_repeats):
						out = self.run(pipeline = command[command_name])
				elif command_split[0]=="if":
					bool_value = self.run(pipeline = [command_split[1]])
					if bool_value in command[command_name]:
						out = self.run(pipeline = command[command_name][bool_value])
				else:
					print("!!!SPECIAL COMMAND NOT RECOGNIZED!!!")
			else:
				if isinstance(args,list):
					for i,elem in enumerate(args): #handle references
						if "~" in elem:
							args[i] = handle_reference(self, elem, "~")
				elif isinstance(args,str):
					if "~" in args:
						args = handle_reference(self, args, "~")
				else:
					print("!!!ARGS TYPE NOT RECOGNIZED!!!")

				if isinstance(kwargs,dict):
					for key,value in enumerate(kwargs.copy()): #handle references
						if "~" in value:
							kwargs[key] = handle_reference(self, value, "~")
				elif isinstance(kwargs,str):
					if "~" in kwargs:
						kwargs = handle_reference(self, kwargs, "~")
				else:
					print("!!!KWARGS TYPE NOT RECOGNIZED!!!")

				if hasattr(self, command_name):
					out = self.get_composite_key(command_name)(*args, **kwargs)
					if out is not None and len(outs)>0:
						self.setattrs(outs,out)
				else:
					print("COMMAND NOT DEFINED")
		return out

	def get_key_if_exists(self, dct, key, error_value = None):
		try:
			return dct[key]
		except KeyError:
			return error_value

	def get_composite_key(self, relative_key):
		keys = relative_key.split(".")
		value = getattr(self,keys[0])
		for key in keys[1:]:
			value = value[key]
		return value

	def set_composite_key(self, relative_key, set_value):
		keys = relative_key.split(".")
		value = getattr(self,keys[0])
		for i,key in enumerate(keys[1:-1]):
			value = value.setdefault(key, {})
		value[keys[-1]] = set_value

	def load_packages(self, package_name): #IMPORT MODEL-SPECIFIC SCRIPTS
		self.packages = {}
		try:
			self.packages[package_name] = __import__(package_name)
			print("PACKAGE IMPORTED CORRECTLY")
		except ModuleNotFoundError:
			print("PACKAGE", package_name, "NOT FOUND;","CONTINUE PIPELINE")

	def load_data(self, *args, **kwargs):
		return util_data.load_data(*args, **kwargs)

	def setattrs(self, attrs, values):
		for key,value in zip(attrs,values):
			self.set_composite_key(key, value)

	def check_experiment(self):
		experiments_file = os.path.join(self.get_out_folder("exp"), self.exp["name"]+"_exp_list.jsonl")
		if not os.path.isfile(experiments_file):
			print("EXPERIMENT FILE NOT FOUND: INITIALIZE IT")
			open(experiments_file, 'w').close()

			self.exp["experiment_id"] = 0
		else:
			found = False
			with open(experiments_file, "r") as f:
				for experiment_id,row in enumerate(f):
					if self.cfg == row:
						found = True
						break
			
			if found and not self.exp["rewrite"]:
				return False
			else:
				return True

	def get_out_folder(self, out_type):
		return os.path.join(self.exp["prj_fld"],'out',out_type)





	'''	
	def set_model_specific_parameters(self):
		if self.cfg["model"]["name"] == "nn":
			self.cfg["model"]["training"] = self.packages["nn"].set_compiler_parameters(self.cfg["model"]["training"])
			self.cfg["model"]["compiler"] = self.packages["nn"].set_training_parameters(self.cfg["model"]["compiler"])

	def initialize(self):
		#self.load_data()
		#self.adjust_model_parameters()
		#if self.cfg["model"]["name"] == "nn":
		#	self.adjust_training_parameters()
		#	self.adjust_compiler_parameters()

		self.create_folders()
		self.prepare_experiments()

		if self.cfg["model"]["name"] == "nn":
			self.train_dataset = self.specific_packages["nn"].prepare_nn_data() #DO only for NNs ----> CHECK LATER

		self.set_metrics()

	def set_metrics(self):
		if "metrics" in self.cfg["pipeline"]:
			self.computed_metrics = {}
			for metric in self.cfg["pipeline"]["metrics"]:
				self.computed_metrics[metric] = []
	
	def adjust_model_parameters(self):
		if "units_per_layer" in self.cfg["model"]["params"]:
			self.cfg["model"]["params"]["units_per_layer"].append(self.train_y.shape[-1])

	def adjust_training_parameters(self):
		if "class_weight" in self.cfg["model"]["training"]: #COMPUTE CLASS WEIGHT IF REQUESTED
			if self.cfg["model"]["training"]["class_weight"]:
				if len(self.train_y.shape)>1:
					if self.train_y.shape[-1]!=1:
						class_y = np.where(self.train_y)[1]
					else:
						class_y = self.train_y[:,0]
				else:
					class_y = self.train_y
				class_weight = compute_class_weight('balanced',classes = np.unique(class_y),y = class_y)
				self.cfg["model"]["training"]["class_weight"] = dict(enumerate(class_weight))

	def adjust_compiler_parameters(self):
		#set entropy: binary if last units=1, else categorical)
		if self.cfg["model"]["compiler"]["loss"] == "entropy":
			self.cfg["model"]["compiler"]["loss"] = ['binary_crossentropy','categorical_crossentropy'][self.cfg["model"]["params"]["units_per_layer"][-1]>1]

	def old_run(self):
		for repeat_k in range(self.cfg["pipeline"]["num_repeats"]):
			print("REPEAT:", repeat_k)
			for metric in self.computed_metrics:
				self.computed_metrics[metric].append([])

			for command in self.cfg["pipeline"]["pipeline_order"]:
				#print(command)
				#print(self.pipeline_parameters["pipeline_order"])
				
				while command:
					#print("EX:",command)
					in_execution = False
					
					if type(command) is not str:
						secondary_commands = command[1:]
						command = command[0]
					else:
						secondary_commands = ()
					
					if command == "create_model":
						self.create_model()
					
					elif command == "load_trained":
						if not hasattr(self,"model"):
							secondary_commands = ("create_model",command,*secondary_commands)
						else:
							if command == "load_trained":
								weights_loaded = self.load_model_weights(repeat_k) #only for nn??? CHANGE LATER
								if weights_loaded:
									secondary_commands = ()

					elif command == "train_model":
						self.train_model()
		
					elif command == "compute_metrics":
						self.compute_metrics()

					elif command == "save_metrics":
						self.save_metrics()

					elif command == "save_all":
						self.save_all(repeat_k)

					else:
						raise NotImplementedError("PROGRAM NOT IMPLEMENTED FOR COMMAND:", command)

					command = secondary_commands
			
			self.cfg["exp"]["seed"] += 1
		#self.data_parameters["selected_seed"] = original_seed

	def create_folders(self):
		for out_type in ["experiments","img","models","results"]: #,"obj"
			dir_path = self.get_out_folder(out_type)

			if not os.path.isdir(dir_path):
				os.makedirs(dir_path)
				warnings.warn("created folder : " + dir_path)

	def create_model(self):
		if self.model["name"] == "nn":
			if self.model["backend"] == "tf":
				self.packages[package_name].compat.v1.set_random_seed(self.exp["seed"])
			
				compiler_parameters = deepcopy(self.model["compiler"])
				compiler_parameters["optimizer"] = self.model["compiler"]["optimizer"]

			self.model = self.specific_packages[self.cfg["model"]["name"]].nn_model(self.train_x.shape[1:], compiler_parameters = compiler_parameters, **self.cfg["model"]["params"])

			if self.cfg["exp"]["verbose"]>0:
				print(self.model.summary())
		else:
			raise NotImplementedError("MODEL TYPE NOT IMPLEMENTED FOR", self.cfg["model"]["name"])

	def train_model(self):
		start_time = datetime.now()
		if self.cfg["model"]["name"] == "nn":
			training_parameters = deepcopy(self.cfg["model"]["training"])
			if "callbacks" in training_parameters:
				training_parameters["callbacks"] = [callback() for callback in training_parameters["callbacks"]]
			print(training_parameters)

			if (np.prod(self.train_x.shape)/8)<3e3: #run on CPU
				if self.cfg["exp"]["verbose"]>0:
					print(("*"*5)+"TRAIN ON CPU"+("*"*5))
				#tf.config.set_visible_devices([], 'GPU')
				with tf.device('/cpu:0'):
					history = self.model.fit(self.train_dataset, **training_parameters)
			else: #run on GPU
				if self.cfg["exp"]["verbose"]>0:
					print(("*"*5)+"TRAIN ON GPU"+("*"*5))
				#tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
				with tf.device('/gpu:4'):
					history = self.model.fit(self.train_dataset,**training_parameters)

			# #SAVE LOSS PLOT???
			# plt.plot(history.history['loss'])
			# plt.title('model loss')
			# plt.ylabel('loss')
			# plt.xlabel('epoch')
			# plt.legend(['train'], loc='upper left')
			# plt.show()
		else:
			raise NotImplementedError("MODEL TYPE NOT IMPLEMENTED FOR", self.cfg["model"]["name"])
		
		end_time = datetime.now()

		if self.exp["verbose"]>0:
			print("TRAIN ET:",end_time-start_time)

	def compute_metrics(self):
		self.train_y_pred = self.model.predict(self.train_x)
		self.test_y_pred = self.model.predict(self.test_x)
		for metric in self.computed_metrics:
			#-1 because cosnider last repeat; append because compute_metrics may be called multiple times in a single repeat
			self.computed_metrics[metric][-1].append(compute_metric(metric, self.model,
																	train_x = self.train_x, train_y = self.train_y,
																	train_y_pred = self.train_y_pred,
																	test_x = self.test_x, test_y = self.test_y,
																	test_y_pred = self.test_y_pred))

	def save_experiment_parameters(self, old_experiments):
		with open(os.path.join(self.get_out_folder("experiments"),"experiments_list.pkl"), "wb") as f:
			pkl.dump(old_experiments, f)
			
		with open(os.path.join(self.get_out_folder("experiments"),"experiments_list.json"), "a") as f:
			all_parameters = self.get_all_parameters()
			print(all_parameters)
			json.dump(all_parameters, f)
			f.write(os.linesep)

	def get_experiments_name(self, out_type):
		return os.path.join(self.get_out_folder(out_type),str(self.experiment_id))

	def get_model_name(self, repeat_k):
		name = self.get_experiments_name("models") + "_" + str(repeat_k) + ".h5"
		
		return name

	def load_model_weights(self, repeat_k):
		model_name = self.get_model_name(repeat_k)
		if self.cfg["exp"]["verbose"]>0:
			print("LOADING", model_name)

		if os.path.isfile(model_name):
			self.model.load_weights(model_name)
			return True
		else:
			warnings.warn("CANNOT LOAD MODEL WEIGHTS: FILE NOT FOUND")
			return False

	def save_all(self, repeat_k):
		self.save_model(repeat_k)
		self.save_metrics()

	def save_model(self, repeat_k):
		filename = self.get_model_name(repeat_k)
		self.model.save_weights(filename)

	def save_metrics(self):
		filename = self.get_experiments_name("results") + ".pkl"
		with open(filename, 'wb') as f:
			pkl.dump(self.computed_metrics, f)
	'''