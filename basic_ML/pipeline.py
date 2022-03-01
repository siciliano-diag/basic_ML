#IMPORTS
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import json

import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
#from tensorflow.keras.models import load_model

from util_data import *
from util_training import *
from util_metrics import *

from copy import deepcopy


class Pipeline():
  def __init__(self, pipeline_parameters, data_parameters, model_parameters, compiler_parameters = {}, training_parameters = {}):
    self.set_pipeline_parameters(pipeline_parameters)
    self.set_data_parameters(data_parameters)
    self.set_model_parameters(model_parameters)
    self.set_compiler_parameters(compiler_parameters)
    self.set_training_parameters(training_parameters)

    self.load_model_specific_libraries()

    self.initialize()

  def set_pipeline_parameters(self, pipeline_parameters):
    self.pipeline_parameters = deepcopy(pipeline_parameters)

    if "metrics" in self.pipeline_parameters:
      self.computed_metrics = {}
      for metric in self.pipeline_parameters["metrics"]:
        self.computed_metrics[metric] = []

  def set_data_parameters(self, data_parameters):
    self.data_parameters = deepcopy(data_parameters)

    if "cwd" not in self.pipeline_parameters:
      self.pipeline_parameters["cwd"] = ""


  def set_model_parameters(self, model_parameters):
    self.model_parameters = deepcopy(model_parameters)

    #if hasattr(self,"train_y"): #CHANGE LAST LAYER UNITS
    #  self.adjust_model_parameters()

    if hasattr(self,"model"):
      del self.model

  def set_training_parameters(self, training_parameters):
    self.verbose = training_parameters["verbose"]
    self.training_parameters = deepcopy(training_parameters)

    if "callbacks" in self.training_parameters:
      callbacks = self.training_parameters["callbacks"]

      self.callbacks_initializer = []
      for callback in callbacks:
        if type(callback) is str:
          callback_name = callback
          callback_params = {}
        else:
          callback_name = callback[0]
          callback_params = callback[1]

        if callback_name == "custom":
          self.callbacks_initializer.append(lambda : CustomEarlyStopping(**callback_params))
        else:
          raise NotImplementedError("CALLBACK NOT IMPLEMENTED FOR", callback)

  def set_compiler_parameters(self, compiler_parameters):
    self.compiler_parameters = deepcopy(compiler_parameters)

    if "optimizer" in self.compiler_parameters:
      optimizer = self.compiler_parameters["optimizer"]
      if type(optimizer) is str:
        optimizer_name = optimizer
        optimizer_params = {}
      else:
        optimizer_name = optimizer[0]
        optimizer_params = optimizer[1]

      if optimizer_name == "adam":
        self.optimizer_initializer = lambda : tf.keras.optimizers.Adam(**optimizer_params)
      else:
        raise NotImplementedError("OPTIMIZER NOT IMPLEMENTED FOR", optimizer)

  def initialize(self):
    self.load_data()
    self.adjust_model_parameters()
    self.adjust_training_parameters()
    self.adjust_compiler_parameters()

    self.create_folders()
    self.prepare_experiments()

    self.prepare_data() #DO only for NNs ----> CHECK LATER

  def prepare_experiments(self):
    print("FIND EXPERIMENT")
    experiments_path = os.path.join(self.get_out_folder("experiments"),"experiments_list.pkl")
    all_parameters = self.get_all_parameters()
    if not os.path.isfile(experiments_path):
      self.experiment_id = 0
      old_experiments = {}
    else:
      
      with open(experiments_path, "rb") as f:
        old_experiments = pkl.load(f)
        #print(old_experiments)
      
        for self.experiment_id,line_parameters in old_experiments.items():
          if line_parameters == all_parameters:
            print("FOUUUUUND")
            return None
          
      #print("LC",line_cont)
      self.experiment_id += 1
    old_experiments[self.experiment_id] = all_parameters
    self.save_experiment_parameters(old_experiments)

  def load_model_specific_libraries(self):
    #IMPORT MODEL-SPECIFIC SCRIPTS
    if self.model_parameters["model_type"] in ["nn","newron"]:
      if self.model_parameters["model_type"] == "newron":  #NEWRON
        #try:
          global newron
          newron = __import__('newron')

          global newron_decoding
          newron_decoding = __import__('newron_decoding')
        #except ImportError:
        #  raise ImportError('NEWRON NOT FOUND')
      elif self.model_parameters["model_type"] == "nn": #FEED FORWARD NN
        global nn
        nn = __import__('nn')
    else:
      raise NotImplementedError("MODEL TYPE NOT IMPLEMENTED FOR", self.model_parameters["model_type"])

  def prepare_data(self):
    if self.model_parameters["model_type"] in ["nn","newron"]: #if NN: use tf Dataset
      self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
      self.train_dataset = self.train_dataset.cache().batch(self.data_parameters["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)

  def adjust_model_parameters(self):
    if "units_per_layer" in self.model_parameters:
      self.model_parameters["units_per_layer"].append(self.train_y.shape[-1])
    
    '''
    #CHANGE LAST UNITS
    if self.train_y.shape[-1] != self.model_parameters["units_per_layer"][-1]:
      warnings.warn("WARNING: LAST LAYER UNITS DIFFERENT FROM Y SHAPE --> FORCING CHANGE OF UNITS")
      self.model_parameters["units_per_layer"][-1] = self.train_y.shape[-1]
    '''

  def adjust_training_parameters(self):
    if "class_weight" in self.training_parameters: #COMPUTE CLASS WEIGHT IF REQUESTED
      if self.training_parameters["class_weight"]:
        if len(self.train_y.shape)>1:
          if self.train_y.shape[-1]!=1:
            class_y = np.where(self.train_y)[1]
          else:
            class_y = self.train_y[:,0]
        else:
          class_y = self.train_y
        class_weight = compute_class_weight('balanced',
                                          classes = np.unique(class_y),
                                          y = class_y)
        self.training_parameters["class_weight"] = dict(enumerate(class_weight))

  def adjust_compiler_parameters(self):
    #set entropy: binary if last units=1, else categorical)
    if self.compiler_parameters["loss"] == "entropy":
      self.compiler_parameters["loss"] = ['binary_crossentropy','categorical_crossentropy'][self.model_parameters["units_per_layer"][-1]>1]
    
    '''
    elif loss=="rank":
      loss = tfr.losses.make_loss_fn(
              "pairwise_logistic_loss",
              lambda_weight=tfr.losses_impl.DCGLambdaWeight(
                  gain_fn=(lambda y: 2**y - 1),
                  rank_discount_fn=(lambda r: 1. / tf.math.log1p(r)),
                  normalized=True,
                  smooth_fraction=1.
              )
          )
    '''
    #print("LOSS:",loss)

  def run(self):
    #original_seed = self.data_parameters["selected_seed"]
    for repeat_k in range(self.pipeline_parameters["num_repeats"]):
      print("REPEAT:", repeat_k)
      for metric in self.computed_metrics:
        self.computed_metrics[metric].append([])

      for command in self.pipeline_parameters["pipeline_order"]:
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

          elif command == "decode_all":
            if self.model_parameters["model_type"] == "newron":
              self.decode_plot()
              self.decode_suff_nec()
    
          elif command == "compute_metrics":
            self.compute_metrics()

          elif command == "save_metrics":
            self.save_metrics()

          elif command == "save_all":
            self.save_all(repeat_k)

          else:
            raise NotImplementedError("PROGRAM NOT IMPLEMENTED FOR COMMAND:", command)

          command = secondary_commands
      
      self.data_parameters["selected_seed"] += 1
    #self.data_parameters["selected_seed"] = original_seed

  def load_data(self):
    self.train_x, self.train_y, self.test_x, self.test_y, self.x_scaler, self.min_max_per_feat = load_data(**self.data_parameters)

  def create_folders(self):
    for out_type in ["experiments","img","models","results"]: #,"obj"
      dir_path = self.get_out_folder(out_type)

      if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        warnings.warn("created folder : " + dir_path)

  def create_model(self):
    #CREATE NN MODEL
    if self.model_parameters["model_type"] in ["nn","newron"]:
      tf.compat.v1.set_random_seed(self.data_parameters["selected_seed"]) #SET SEED
      
      compiler_parameters = deepcopy(self.compiler_parameters)
      compiler_parameters["optimizer"] = self.optimizer_initializer()

      if self.model_parameters["model_type"] == "newron":  #NEWRON
        self.model = newron.newron_model(self.train_x.shape[1:], self.min_max_per_feat, compiler_parameters = compiler_parameters, **self.model_parameters)
      elif self.model_parameters["model_type"] == "nn": #FEED FORWARD NN
        self.model = nn.nn_model(self.train_x.shape[1:], compiler_parameters = compiler_parameters, **self.model_parameters)

      if self.verbose>0:
        print(self.model.summary())
    else:
      raise NotImplementedError("MODEL TYPE NOT IMPLEMENTED FOR", self.model_parameters["model_type"])

  def train_model(self):
    start_time = datetime.now()
    if self.model_parameters["model_type"] in ["nn","newron"]:
      training_parameters = deepcopy(self.training_parameters)
      training_parameters["callbacks"] = [callback() for callback in self.callbacks_initializer]

      if (np.prod(self.train_x.shape)/8)<3e3: #run on CPU
        if self.verbose>0:
          print(("*"*5)+"TRAIN ON CPU"+("*"*5))
        #tf.config.set_visible_devices([], 'GPU')
        with tf.device('/cpu:0'):
          history = self.model.fit(self.train_dataset, **training_parameters)
      else: #run on GPU
        if self.verbose>0:
          print(("*"*5)+"TRAIN ON GPU"+("*"*5))
        #tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
        with tf.device('/gpu:4'):
          history = self.model.fit(self.train_dataset,**training_parameters)

      ''' SAVE LOSS PLOT???
      plt.plot(history.history['loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train'], loc='upper left')
      plt.show()
      '''
    else:
      raise NotImplementedError("MODEL TYPE NOT IMPLEMENTED FOR", self.model_parameters["model_type"])
    
    end_time = datetime.now()

    if self.verbose>0:
      print("TRAIN ET:",end_time-start_time)

  def compute_metrics(self):
    self.train_y_pred = self.model.predict(self.train_x)
    self.test_y_pred = self.model.predict(self.test_x)
    for metric in self.pipeline_parameters["metrics"]:
      #-1 because cosnider last repeat; append because compute_metrics may be called multiple times in a single repeat
      self.computed_metrics[metric][-1].append(compute_metric(metric, self.model,
                                                              train_x = self.train_x, train_y = self.train_y,
                                                              train_y_pred = self.train_y_pred,
                                                              test_x = self.test_x, test_y = self.test_y,
                                                              test_y_pred = self.test_y_pred))
    
  
  def save_experiment_parameters(self, old_experiments):
    with open(os.path.join(self.get_out_folder("experiments"),"experiments_list.pkl"), "wb") as f:
      pkl.dump(old_experiments, f)
      
    '''
    with open(os.path.join(self.get_out_folder("experiments"),"experiments_list.json"), "a") as f:
      all_parameters = self.get_all_parameters()
      print(all_parameters)
      json.dump(all_parameters, f)
      f.write(os.linesep)
    '''

  def get_all_parameters(self):
    all_parameters = {"pipeline_parameters": self.pipeline_parameters,
                      "data_parameters": self.data_parameters,
                      "model_parameters":self.model_parameters,
                      "compiler_parameters": self.compiler_parameters,
                      "training_parameters":self.training_parameters}

    return all_parameters

  def get_out_folder(self, out_type):
    name = os.path.join(self.data_parameters["cwd"],'..','out',out_type,
                        self.data_parameters["dataname"])

    return name

  def get_experiments_name(self, out_type):
    name = os.path.join(self.get_out_folder(out_type),str(self.experiment_id))
    
    return name

  def get_model_name(self, repeat_k):
    name = self.get_experiments_name("models") + "_" + str(repeat_k) + ".h5"
    
    '''
    name = str(self.dataname+"/model_units"+"-".join([str(x) for x in self.model_parameters["units_per_layer"]])+
                                                        "_inpf-"+self.model_parameters["input_function"]+
                                                        "_aggf-"+self.model_parameters["aggregation_function"]+
                                                        "_outf-"+self.model_parameters["output_function"])
    '''

    return name

  def decode_suff_nec(self):
    input_min_max = self.min_max_per_feat
    img_folder = self.get_experiments_name("img") + "_" + str(repeat_k)
    if not os.path.isdir(img_folder):
      os.makedirs(img_folder)
      warnings.warn("created folder : " + img_folder)

    prec = 1000
    
    newron_decoding.decode_suff_nec(self.model.layers, input_min_max, self.x_scaler, prec, img_folder)
  
  def decode_plot(self, repeat_k):
    input_min_max = self.min_max_per_feat
    img_folder = self.get_experiments_name("img") + "_" + str(repeat_k)
    if not os.path.isdir(img_folder):
      os.makedirs(img_folder)
      warnings.warn("created folder : " + img_folder)

    prec = 1000
    
    newron_decoding.decode_plot_general(self.model.layers, input_min_max, self.x_scaler, prec, img_folder)
  
  def load_model_weights(self, repeat_k):
    model_name = self.get_model_name(repeat_k)
    if self.verbose>0:
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
  def train_tree(self):
    self.clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    self.clf = self.clf.fit(self.train_x, self.train_y)

    #Predict the response for test dataset
    self.tree_train_y_pred = self.clf.predict(self.train_x)
    self.tree_test_y_pred = self.clf.predict(self.test_x)

  def load_model(self):
    #model_name = '../out/models/'+self.dataname+"/model_splits"+"-".join([str(x) for x in self.num_splits_per_var])+"_units"+"-".join([str(x) for x in self.units_per_layer])

    model_name = str('../out/models/' + self.get_model_name())

    #print(model_name)

    if os.path.isdir(model_name):
      self.model = tf.keras.models.load_model(model_name)
    else:
      if self.verbose>0:
        print("CANNOT LOAD MODEL: FILE NOT FOUND")
  '''