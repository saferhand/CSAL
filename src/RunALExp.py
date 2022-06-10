import pandas as pd
from river import stream
import os
from ActiveLearning import EvaluatorAL
from ActiveLearning import Random

def run_experiments_one_pm(data_folder, data_list, compared_method_list, parallel_experiment_times, output_folder,parval):

    tempname = 'r' + str(parval[0])

    for arff_data in data_list:
        for algorithm in compared_method_list:
            for times in range(0, parallel_experiment_times):

                result_csv_name = arff_data + '_'+ algorithm + tempname + str(times) + '_0.csv'
                print(result_csv_name)

                path = data_folder + arff_data
                X_y = stream.iter_arff(path, target='class')

                expmodel = Random(X_y,
                                    parval[0],
                                    parval[1]
                )

                labellist, predlist, allist, ddlist  = expmodel.learning_procedure()
                eva = EvaluatorAL()
                resulfdf = eva.evaluate_accuracy_score(labellist, predlist, allist, 100)
                resulfdf.to_csv(output_folder + result_csv_name, index= False)

parameterslist = []
r_list = [0.3]
i_list = [100]

for rvalue in r_list:
    for ival in i_list:
        parameterslist.append([rvalue, ival])

data_folder = ''
data_list = os.listdir(data_folder)

parallel_experiment_times = 10
output_folder = ''
compared_method_list =[
    'Randompy'
]

for pa in parameterslist:
    run_experiments_one_pm(data_folder, data_list, compared_method_list, parallel_experiment_times, output_folder, pa)
