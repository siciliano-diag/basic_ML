#IMPORTS
from copy import deepcopy
import types
import os

def set_to_self_methods(self, module):
	#module = __import__(".", fromlist=[package_name])
	for method_name in dir(module):
		method = getattr(module,method_name)
		if isinstance(method, types.FunctionType):
			setattr(self,method_name,types.MethodType(getattr(module,method_name),self))

def set_cfg_to_self(self):
	for key, value in self.cfg.items():
		setattr(self, key, deepcopy(value))
	
	if not hasattr(self, 'verbose'):
		self.verbose = False

#def prepare_experiments(self):
	#if "seed" not in self.cfg["exp"]:
	#	self.cfg["exp"]["seed"] = 42

	#if "verbose" not in self.cfg["exp"]:
	#	self.cfg["exp"]["verbose"] = 0
	
def get_key_if_exists(self, dct, key, error_value = None):
	try:
		return dct[key]
	except KeyError:
		return error_value

def get_composite_attr(self, relative_key):
	keys = relative_key.split(".")
	value = getattr(self,keys[0])
	for key in keys[1:]:
		value = getattr(value,key)
	return value

def get_composite_key(self, relative_key):
	keys = relative_key.split(".")
	value = getattr(self,keys[0])
	for key in keys[1:]:
		value = value[key]
	return value

def set_composite_key(self, relative_key, set_value):
	keys = relative_key.split(".")
	value = self
	if len(keys)>1:
		value = getattr(value,keys[0])
		for i,key in enumerate(keys[1:-1]):
			value = value.setdefault(key, {})
		value[keys[-1]] = set_value
	else:
		setattr(value,keys[-1],set_value)

def setattrs(self, attrs, values):
	for key,value in zip(attrs,values):
		self.set_composite_key(key, value)

def get_out_folder(self, out_type):
	out_folder = os.path.join(self.exp["prj_fld"],'out',out_type)
	if not os.path.isdir(out_folder):
		print(out_folder,"not found --> creating")
		os.makedirs(out_folder)
	return out_folder