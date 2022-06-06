#IMPORTS
import os
import json

from . import util_data
from . import util_cfg

def load_packages(self, *package_names): #IMPORT MODEL-SPECIFIC SCRIPTS
	for package_name in package_names:
		print("IMPORTING",package_name)
		#try:
			# if os.path.isdir(package_name):
		setattr(self,package_name,__import__(package_name))
		print("PACKAGE IMPORTED CORRECTLY")

			# 	print("IMPORTING SUBMODULES")
			# 	load_packages(getattr(self,package_name), *[os.path.join(package_name,x) for x in os.listdir(package_name)])
			# else:
			# 	if package_name[-3:] == ".py":
			# 		setattr(self,package_name,__import__(package_name[:-3]))
			# 		print("PACKAGE IMPORTED CORRECTLY")
			# 	else:
			# 		print("FILE NOT RECOGNIZED")
		#except ModuleNotFoundError:
		#	print("PACKAGE", package_name, "NOT FOUND;", "CONTINUE PIPELINE")

def load_data(self, *args, **kwargs):
	return util_data.load_data(*args, **kwargs)

def get_experiment_id(self, cfg = None, experiment_name = None):
	cfg = self.cfg if cfg is None else cfg
	experiment_name = self.exp["name"] if experiment_name is None else experiment_name
	
	experiments_file = os.path.join(self.get_out_folder("exp"), experiment_name+"_exp_list.jsonl")
	experiment_id = 0
	tot_experiments = -1
	exp_found = False
	if os.path.isfile(experiments_file):
		with open(experiments_file, "r") as f:
			for tot_experiments,row in enumerate(f):
				row_cfg = util_cfg.ConfigObject(json.loads(row))
				if cfg == row_cfg:
					exp_found = True
					experiment_id = tot_experiments
					if experiment_id!=0:
						print("!!!SAME CONFIG MULTIPLE TIMES?!!!")
	
	tot_experiments += 1
	if not exp_found:
		experiment_id = tot_experiments
	
	return exp_found, experiment_id, tot_experiments

def get_set_experiment_id(self, cfg=None, experiment_name = None):
	cfg = self.cfg if cfg is None else cfg
	experiment_name = self.exp["name"] if experiment_name is None else experiment_name

	exp_found, self.exp["experiment_id"], self.exp["tot_experiments"] = get_experiment_id(self, cfg, experiment_name)
	return exp_found

def save_experiment(self, out_folder = None, experiment_name = None, experiment_id = None, tot_experiments = None, cfg = None):
	out_folder = self.get_out_folder("exp") if out_folder is None else out_folder
	experiment_name = self.exp["name"] if experiment_name is None else experiment_name
	experiment_id = self.exp["experiment_id"] if experiment_id is None else experiment_id
	tot_experiments = self.exp["tot_experiments"] if tot_experiments is None else tot_experiments
	cfg = self.cfg if cfg is None else cfg

	experiments_file = os.path.join(out_folder, experiment_name+"_exp_list.jsonl")
	if not os.path.isfile(experiments_file):
		print("EXPERIMENT FILE NOT FOUND: INITIALIZE IT")
		open(experiments_file, 'w').close()

	if experiment_id == tot_experiments: #NEW_EXPERIMENTS
		with open(experiments_file,'a') as f:
			json.dump(cfg,f)
			f.write("\n")

	'''
	else: #REPLACE EXPERIMENT
		lines = open(experiments_file, 'r').readlines()
		lines[line_num] = text
		out = open(file_name, 'w')
		out.writelines(lines)
		out.close()
	'''