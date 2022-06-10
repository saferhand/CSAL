import pandas as pd
from river import stream
import os
from OATL import EvaluatorV0
from CSAL import CSAL

def run_experiments_one_pm(data_folder, data_list, compared_method_list, parallel_experiment_times, output_folder,parval):

    tempname = 'r' + str(parval[0]) + \
               'u' + str(parval[1]) + \
               'c' + str(parval[2]) + \
               'i' + str(parval[3]) + \
               'd' + str(parval[4]) + \
               'w' + str(parval[5]) + \
               's' + str(parval[6])

    for arff_data in data_list:
        for algorithm in compared_method_list:
            for times in range(0, parallel_experiment_times):

                result_csv_name = arff_data + '_'+ algorithm + tempname + '_ps1_rpetime' + str(times) + '_0.csv'
                print(result_csv_name)

                path = data_folder + arff_data
                X_y = stream.iter_arff(path, target='class')

                expmodel = CSAL(X_y,
                                    parval[0],
                                    parval[1],
                                    parval[2],
                                    parval[3],
                                    parval[4],
                                    parval[5],
                                    parval[6]
                )

                labellist, predlist, allist, ddlist, declist, problist, alslist  = expmodel.learning_procedure()

                andsimlist = []
                orsimlist = []
                print(sum(declist))
                for i in range(1, len(declist) + 1):
                    if declist[i - 1] and problist[i - 1]:
                        andsimlist.append(1)
                    else:
                        andsimlist.append(0)

                    if declist[i - 1] or problist[i - 1]:
                        orsimlist.append(1)
                    else:
                        orsimlist.append(0)

                declistsum = []
                problistsum = []
                andsimlistsum = []
                orsimlistsum = []

                for i in range(1, len(declist) + 1):
                    declistsum.append(sum(declist[0:i]))
                    problistsum.append(sum(problist[0:i]))
                    andsimlistsum.append(sum(andsimlist[0:i]))
                    orsimlistsum.append(sum(orsimlist[0:i]))


                # declist = pd.DataFrame(declist)
                # declist.to_csv(arff_data + "declist.csv")
                # declistsum = pd.DataFrame(declistsum)
                # declistsum.to_csv(arff_data +"declistsum.csv")
                # problist = pd.DataFrame(problist)
                # problist.to_csv(arff_data + "problist.csv")
                # problistsum = pd.DataFrame(problistsum)
                # problistsum.to_csv(arff_data + "problistsum.csv")
                # 临时的数据分析
                newdf = []
                newdf = pd.DataFrame(newdf)
                newdf['declistsum'] = declistsum
                newdf['problistsum'] = problistsum
                newdf['orsum'] = orsimlistsum
                newdf['andsum'] = andsimlistsum
                newdf.to_csv('C:/Users/C2/Github/CSOATL/0Algorithms/PythonCode/Tempoutputs/' + arff_data + "SimlarAna.csv")

                eva = EvaluatorV0()
                resulfdf = eva.evaluate_accuracy_score(labellist, predlist, allist, alslist, 100)
                resulfdf.to_csv(output_folder + result_csv_name, index= False)

parameterslist = []
r_list = [0.05, ]
u_list = [0.1]
c_list = [0.8]
i_list = [50]
d_list = [20]
w_list = [0.995]
s_list = [1]

for rvalue in r_list:
    for uval in u_list:
        for cval in c_list:
            for ival in i_list:
                for dval in d_list:
                    for wval in w_list:
                        for sval in s_list:
                            parameterslist.append([rvalue, uval, cval, ival, dval, wval, sval])

print(parameterslist)

exit()

data_folder = '' # Set
data_list = os.listdir(data_folder)

parallel_experiment_times = 1
output_folder = '' # Set
compared_method_list =[
    'CSOATL'
]

for pa in parameterslist:
    run_experiments_one_pm(data_folder, data_list, compared_method_list, parallel_experiment_times, output_folder, pa)
