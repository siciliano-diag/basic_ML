#IMPORTS
import types
import copy

from . import util_cfg
from . import util_pipeline #import set_to_self_methods
from . import pipeline_methods

class Pipeline():
	def __init__(self):
		util_pipeline.set_to_self_methods(self, util_pipeline)
		util_pipeline.set_to_self_methods(self, pipeline_methods)
		util_pipeline.set_to_self_methods(self, model_utils)

		self.cfg = util_cfg.load_configuration()

		self.set_cfg_to_self()
		for key in ["exp","pipeline"]: #Pipeline and exp are not part of the config
			del self.cfg[key] 

		if self.verbose:
			print(self.cfg)

	def run(self, start_cmd = 0, end_cmd = None, pipeline = None):
		pipeline = self.pipeline if pipeline is None else pipeline

		out = None
		for command in pipeline[start_cmd:end_cmd]:
			print("EXECUTING:", command)
			if isinstance(command,str):
				command_name, args, kwargs, outs = command, [], {}, []
			elif isinstance(command,dict):
				command_name = list(command.keys())[0]
				if  isinstance(command[command_name],dict):
					args = self.get_key_if_exists(command[command_name],"args",[]).copy()
					kwargs = self.get_key_if_exists(command[command_name],"kwargs",{})
					if isinstance(kwargs,dict):
						kwargs = kwargs.copy()
					outs = self.get_key_if_exists(command[command_name],"outs",[]).copy()
				else:
					args, kwargs, outs = [], {}, []
			else:
				print("ERROR: command not str or dict")

			if " " in command_name: #check if special words
				command_split = command_name.split(" ")
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
					if not hasattr(self,repeats):
						self.repeats = {}
					if len(command_split)>2:
						repeat_key = command_split[2]
					else:
						repeat_key = len(self.repeats)

					for repeat_i in range(num_repeats):
						self.repeats[repeat_key] = repeat_i
						out = self.run(pipeline = command[command_name])
					self.repeats.pop(repeat_key, None)
				elif command_split[0]=="if":
					bool_value = self.run(pipeline = [command_split[1]])
					if bool_value in command[command_name]:
						out = self.run(pipeline = command[command_name][bool_value])
				else:
					print("!!!SPECIAL COMMAND NOT RECOGNIZED!!!")
			else:
				if isinstance(args,list):
					for i,elem in enumerate(args): #handle references
						if isinstance(elem,str):
							if "~" in elem:
								args[i] = util_cfg.handle_reference(self, elem, "~")
				elif isinstance(args,str):
					if "~" in args:
						args = util_cfg.handle_reference(self, args, "~")
					args = [args]
				else:
					print("!!!ARGS TYPE NOT RECOGNIZED!!!")

				if isinstance(kwargs,dict):
					for key,value in kwargs.copy().items(): #handle references
						if isinstance(value,str):
							if "~" in value:
								kwargs[key] = util_cfg.handle_reference(self, value, "~")
				elif isinstance(kwargs,str):
					if "~" in kwargs:
						kwargs = util_cfg.handle_reference(self, kwargs, "~")
				else:
					print("!!!KWARGS TYPE NOT RECOGNIZED!!!")

				# if hasattr(self, command_name):
				out = self.get_composite_attr(command_name)(*args, **kwargs)
				if out is not None and len(outs)>0:
					self.setattrs(outs,out)
				# else:
				# 	print("COMMAND NOT DEFINED")
		return out