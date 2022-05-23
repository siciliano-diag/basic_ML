#IMPORTS
import os

from . import util_data

def load_packages(self, *package_names): #IMPORT MODEL-SPECIFIC SCRIPTS
	for package_name in package_names:
		print("IMPORTING",package_name)
		try:
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
		except ModuleNotFoundError:
			print("PACKAGE", package_name, "NOT FOUND;", "CONTINUE PIPELINE")

def load_data(self, *args, **kwargs):
	return util_data.load_data(*args, **kwargs)

def check_experiment(self):
	experiments_file = os.path.join(self.get_out_folder("exp"), self.exp["name"]+"_exp_list.jsonl")
	if not os.path.isfile(experiments_file):
		self.exp["experiment_id"] = 0
	else:
		found = False
		experiment_id = 0
		i = -1
		with open(experiments_file, "r") as f:
			for i,row in enumerate(f):
				if self.cfg == row:
					found = True
					experiment_id = i
					if experiment_id!=0:
						print("!!!SAME CONFIG MULTIPLE TIMES?!!!")
		self.exp["tot_experiments"] = i+1
		if found:
			self.exp["experiment_id"] = experiment_id
			return True
		else:
			self.exp["experiment_id"] = self.exp["tot_experiments"]
	return False

def save_experiment(self):
	experiments_file = os.path.join(self.get_out_folder("exp"), self.exp["name"]+"_exp_list.jsonl")
	if not os.path.isfile(experiments_file):
		print("EXPERIMENT FILE NOT FOUND: INITIALIZE IT")
		open(experiments_file, 'w').close()

	if self.exp["experiment_id"] == self.exp["tot_experiments"]: #NEW_EXPERIMENTS
		with open(experiments_file,'a') as f:
			f.write(self.cfg)
	'''
	else: #REPLACE EXPERIMENT
		lines = open(experiments_file, 'r').readlines()
		lines[line_num] = text
		out = open(file_name, 'w')
		out.writelines(lines)
		out.close()
	'''
	