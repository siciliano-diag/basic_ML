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

def set_training_parameters(training_parameters):
	if "callbacks" in training_parameters:
		callbacks = training_parameters

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
		
def set_compiler_parameters(compiler_parameters):
	if "optimizer" in compiler_parameters:
		optimizer = compiler_parameters["optimizer"]
		if type(optimizer) is str:
			optimizer_name = optimizer
			optimizer_params = {}
		else:
			optimizer_name = optimizer[0]
			optimizer_params = optimizer[1]

		if optimizer_name == "adam":
			compiler_parameters["optimizer"] = lambda : tf.keras.optimizers.Optimizer(compiler_parameters["optimizer"])
		else:
			raise NotImplementedError("OPTIMIZER NOT IMPLEMENTED FOR", optimizer)

def prepare_nn_data(self, data_parameters):
	train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	train_dataset = train_dataset.cache().batch(self.cfg["data"]["batch_size"]).prefetch(tf.data.experimental.AUTOTUNE)
