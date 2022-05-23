import yaml
import os
import numpy as np
import re
import argparse

def load_configuration():
	args = parse_arguments()
	
	config_path = args.pop('config_path', None)
	config_name = args.pop('config_name', None)

	cfg = load_yaml(config_path, config_name)

	cfg = handle_globals(cfg)

	cfg = handle_relatives(cfg, cfg)

	for k,v in args.items():
		cfg.set_composite_key(k,v)

	return cfg

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default = "../cfg")
	parser.add_argument("--config_name", default = "config")
	parsed, unknown = parser.parse_known_args()
	for arg in unknown:
		if arg.startswith(("-", "--")):
			parser.add_argument(arg.split('=')[0])
	args = parser.parse_args()
	return vars(args)

def load_yaml(config_path, config_name):
	with open(os.path.join(config_path,config_name+".yaml"), 'r') as f:
		cfg = yaml.safe_load(f)

	cfg = ConfigObject(cfg)

	cfg = handle_additions(cfg, config_path)

	return cfg

def handle_additions(cfg, config_path):
	for key,value in cfg.copy().items():
		if key[0]=="+":
			del cfg[key]
			key = key.replace("+","").strip()
			
			optional,key = check_optional(key)

			#if isinstance(value, list):
			#	value = value[0] #####!!!!! CHANGE TO HANDLE LISTS!!!!!!#####
			try:
				additional_cfg = load_yaml(os.path.join(config_path,key),value)

				cfg = raise_globals(cfg, additional_cfg)

				cfg[key] = additional_cfg
			except FileNotFoundError:
				if not optional:
					raise FileNotFoundError("SPECIALIZED CFG NOT FOUND:"+os.path.join(config_path,key,value))
				else:
					print("SPECIALIZED CFG NOT FOUND, BUT OPTIONAL:",os.path.join(config_path,key,value))
	return cfg

def handle_globals(cfg):
	if "_global_" in cfg:
		cfg = merge_dicts(cfg,cfg["_global_"])

		cfg.pop("_global_",None)

	return cfg

def handle_relatives(obj, global_cfg):
	if isinstance(obj, dict):
		for key,value in obj.items():
			obj[key] = handle_relatives(value,global_cfg)
	elif isinstance(obj, list):
		for i, elem in enumerate(obj):
			obj[i] = handle_relatives(elem, global_cfg)
	elif isinstance(obj, str):
		if "$" in obj:
			return handle_reference(global_cfg, obj)
	return obj

def check_optional(key):
	key_split = key.split(" ")
	try:
		optional_id = key_split.index("optional")
		return True," ".join(key_split[:optional_id] + key_split[optional_id+1:])
	except ValueError:
		return False, key

def raise_globals(cfg, new_cfg):
	if "_global_" in new_cfg:
		if "_global_" in cfg:
			cfg["_global_"] = merge_dicts(cfg["_global_"],new_cfg["_global_"])
		else:
			cfg["_global_"] = new_cfg["_global_"]
		new_cfg.pop("_global_",None)

	return cfg
	
def merge_dicts(a, b, path=None):
	if path is None:
		path = []
	for key in b:
		if key in a:
			if isinstance(a[key], dict) and isinstance(b[key], dict):
				merge_dicts(a[key], b[key], path + [str(key)])
			elif a[key] == b[key]:
				pass # same leaf value
			else:
				raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
		else:
			a[key] = b[key]
	return a

def handle_reference(cfg, obj, char="$"):
	matches = [match for match in re.finditer(re.escape(char)+r"\{(.*?)\}",obj)]
	print(matches)
	'''
	if len(matches) == 1:
		match = matches[0]
		start_idx, end_idx = match.span()
		if end_idx-start_idx == len(obj):
			return cfg.get_composite_key(match.group(1))
	'''
	
	new_string = ""
	start_idx, end_idx = 0,-1
	for match in matches:
		span_start, span_end = match.span()
		new_string += obj[start_idx:span_start]
		new_string += cfg.get_composite_key(match.group(1))
		start_idx = span_end
	new_string += obj[start_idx:]
	return new_string

class ConfigObject(dict):
	# dot.notation access to dictionary attributes
	# __getattr__ = dict.__getitem__
	# __setattr__ = dict.__setitem__
	# __delattr__ = dict.__delitem__
	def __init__(self, cfg):
		super().__init__(cfg)
		# for k,v in dct.items():
		# 	if isinstance(v,dict):
		# 		v = dotdict(v)

	def get_composite_key(self, relative_key):
		try:
			value = self
			for key in relative_key.split("."):
				value = value[key]
		except KeyError:
			value = "${"+relative_key+"}"
		return value

	def set_composite_key(self, relative_key, set_value):
		keys = relative_key.split(".")
		value = self
		for i,key in enumerate(keys[:-1]):
			value = value.setdefault(key, {})
		value[keys[-1]] = set_value