#IMPORTS
import os

import util_data

'''
def load_packages(self, package_name): #IMPORT MODEL-SPECIFIC SCRIPTS
	self.packages = {}
	try:
		self.packages[package_name] = __import__(package_name)
		print("PACKAGE IMPORTED CORRECTLY")
	except ModuleNotFoundError:
		print("PACKAGE", package_name, "NOT FOUND;","CONTINUE PIPELINE")
'''

def load_data(self, *args, **kwargs):
	return util_data.load_data(*args, **kwargs)

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