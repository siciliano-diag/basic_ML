#IMPORTS
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def nn_model(input_shape, compiler_parameters = {}, units_per_layer = [1], **kwargs):
  input_layer = Input(shape = input_shape)

  last_layer = input_layer

  for i,units in enumerate(units_per_layer):
    if "layer_args" in kwargs:
      layer_args = kwargs["layer_args"]
    else:
      layer_args = {}
    last_layer = Dense(units, **layer_args)(last_layer)

  model = Model(inputs = input_layer, outputs = last_layer)

  #print(compiler_parameters)
  model.compile(**compiler_parameters) #'adam'

  return model