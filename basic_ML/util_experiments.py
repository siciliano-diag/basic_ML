#IMPORTS
from copy import deepcopy
import numpy as np

def try_combination(pipeline, pipeline_parameters, experiment_parameters, data_parameters, model_parameters, compiler_parameters):
    print("SETTING PARAMS")
    pipeline.set_model_parameters(model_parameters)
    pipeline.adjust_model_parameters()
    pipeline.prepare_experiments()
    pipeline.run()

    print("\n\nRESULTS:", data_parameters["dataname"])
    #print("PCM",pipeline.computed_metrics)
    #print("MEAN0",)
    #print("MEAN1",np.mean(pipeline.computed_metrics[experiment_parameters["metric"]],axis=1)
    
    print(np.mean(pipeline.computed_metrics[experiment_parameters["metric"]],axis=0))
    new_metric = np.mean(pipeline.computed_metrics[experiment_parameters["metric"]],axis=0)[-1][1] #-1 means last computed metrics (in pipeline_order), 1 means test set
    print("NEW METRIC: ", new_metric)
    print("BEST METRIC: ", experiment_parameters["best_metric"])

    experiment_parameters["tested"].append(model_parameters["units_per_layer"])

    if experiment_parameters["metric"] in ["accuracy"]:
        if new_metric>0.999: #or (experiment_parameters["metrics"][0]=="mse" and new_metric<0.01):
            print("PERFECT\n")
        else:
            if (new_metric-experiment_parameters["best_metric"]) < experiment_parameters["eps"]:
                experiment_parameters["patience"] -= 1
                if experiment_parameters["patience"] == 0:
                    return None
            experiment_parameters["best_metric"] = max(new_metric,experiment_parameters["best_metric"])

            for rnd in [1,2]:
                '''
                if rnd==0 and model_parameters[0][0]<5: #increase number of tanh
                    new_model_parameters = deepcopy(model_parameters)
                    for i in range(len(model_parameters[0])):
                        new_model_parameters[0][i] += 1
                    if (data_parameters,new_model_parameters) not in tested:
                        try_comb(data_parameters, new_model_parameters, patience, tested, new_acc)
                '''
                if rnd==1: #increase units
                    for j in range(len(model_parameters["units_per_layer"])):
                        new_model_parameters = deepcopy(model_parameters)
                        new_experiment_parameters = deepcopy(experiment_parameters)
                    #for i in range(len(model_parameters[0])):
                        #new_model_parameters[0][i] *= 2
                        new_model_parameters["units_per_layer"][j] *= 2
                        if new_model_parameters["units_per_layer"] not in experiment_parameters["tested"]:
                            try_combination(pipeline, pipeline_parameters, new_experiment_parameters, data_parameters, new_model_parameters, compiler_parameters)
                elif rnd==2: #increase depth
                    if np.log2(np.sum(model_parameters["units_per_layer"])+2)>=len(model_parameters["units_per_layer"])+1: #len(model_parameters[0])==1 or 
                        new_model_parameters = deepcopy(model_parameters)
                        new_experiment_parameters = deepcopy(experiment_parameters)

                        #new_model_parameters[0] += [new_model_parameters[0][-1]]
                        #new_model_parameters["units_per_layer"][-1]*=2
                        new_model_parameters["units_per_layer"] += [1]#[new_model_parameters[0][-1]]
                        if new_model_parameters["units_per_layer"] not in experiment_parameters["tested"]:
                            try_combination(pipeline, pipeline_parameters, new_experiment_parameters, data_parameters, new_model_parameters, compiler_parameters)
    else:
        raise NotImplementedError("EXPERIMENT NOT IMPLEMENTED FOR METRIC", experiment_parameters["metric"])

    '''
    else:
        if (new_metric-experiment_parameters["best_metric"]) > experiment_parameters["eps"]:
            patience -= 1
            if patience==0:
                return None
        best_acc = min(new_metric,experiment_parameters["best_metric"])
    '''