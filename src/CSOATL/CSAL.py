from river import stream
from river import ensemble
from river import preprocessing
from river import tree
from sklearn.metrics import accuracy_score
from river.drift import HDDM_A
from river.drift import DDM
import collections
import random
import math
from scipy.stats import ks_2samp

class CSAL:

    def __init__(self, datastream,
                 al_rd_thd,
                 al_un_thd,
                 drift_confidence,
                 al_ini_num,
                 al_dd_num,
                 inductive_weight_parameter,
                 asymmetry_weighting_parameter,
                 cs_ratio):

        self.X_y = datastream
        self.AL_RANDOM_THRESHOLD = al_rd_thd
        self.AL_UNCERTAINTY_THRESHOLD = al_un_thd
        self.TARGET_DOMAIN_INDEX = 0
        self.drift_confidence = drift_confidence
        self.al_ini_num = al_ini_num
        self.al_dd_num = al_dd_num
        self.inductive_weight_parameter = inductive_weight_parameter
        self.asymmetry_weighting_parameter = asymmetry_weighting_parameter
        self.cs_ratio = cs_ratio

    def creat_model_for_domain(self):
        model = ensemble.BaggingClassifier(
            model=(
                preprocessing.StandardScaler() |
                tree.HoeffdingTreeClassifier()
            ),
            n_models=10,
            seed=2
        )
        return model

    def creat_ddmodel_for_domain(self):
        ddmodel = HDDM_A()
        return ddmodel

    def get_votes_for_instance(self, x, model):
        y_pred = collections.Counter()
        for bsm in model:
            y_pred.update(bsm.predict_proba_one(x))
        total = sum(y_pred.values())
        if total > 0:
            result = {label: proba / total for label, proba in y_pred.items()}
            return result
        return y_pred


   
    def asymmetric_weighting(self, ms_ensemble, ms_weight, x, y):
        for keys, values in ms_ensemble.items():
            for j, basemodel in enumerate(values): 
                for k, baseclf in enumerate(basemodel):
                    votes = baseclf.predict_proba_one(x)
                    if len(votes) != 2:
                        votes = {0: 0.0, 1: 0.0}
                    if max(votes, key=votes.get) == y:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 + 0.01)
                    else:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 - 0.1)

                        max_prob_index = max(votes, key=votes.get)
                        max_prob = votes.get(max(votes, key=votes.get))
                        votes[max_prob_index] = 0
                        second_prob = votes.get(max(votes, key=votes.get))
                        uncertainty_margin_value = max_prob - second_prob
                        if uncertainty_margin_value >= self.inductive_weight_parameter:
                            ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 - 0.1*self.asymmetry_weighting_parameter)

        return ms_weight


   
    def update_weight(self, ms_ensemble, ms_weight, x, y):
        for keys, values in ms_ensemble.items():
            for j, basemodel in enumerate(values): 
                for k, baseclf in enumerate(basemodel):
                    votes = baseclf.predict_proba_one(x)
                    if len(votes) != 2:
                        votes = {0: 0.0, 1: 0.0}
                    if max(votes, key=votes.get) == y:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 + 0.01)
                    else:
                        ms_weight[keys][j][k] = ms_weight[keys][j][k] * (1 - 0.01)

        return ms_weight

    def cal_eva_similarity_problist(self, x, ms_ensemble):
        temp_pred_for_target_sample = {}
        for source_id, ensemble_classifier_group in ms_ensemble.items():
            temp_pred_for_target_sample[source_id] = []
            for j, baseclf in enumerate(ensemble_classifier_group[-1]): 
                votes = baseclf.predict_proba_one(x)
                temp_pred_for_target_sample[source_id].append(votes)
       
       
       
       
       
       
       
       
       
       
       
       
        pred_prob_list_for_target_sample = {}
        pred_class_list_for_target_sample = {}
        for source_keys, values in temp_pred_for_target_sample.items():
            pred_class_list_for_target_sample[source_keys] = {}
            temp_predclass_list = []
            for baseclf_pred_prob in values:
                if len(baseclf_pred_prob) != 2:
                    temp_predclass_list.append(-1)
                else:
                    temp_predclass_list.append(max(baseclf_pred_prob, key=baseclf_pred_prob.get))
            pred_class_list_for_target_sample[source_keys] = temp_predclass_list

            pred_prob_list_for_target_sample[source_keys] = {}
            for class_label in values[-1].keys():
                temp_predprob_list = []
                for baseclf_pred_prob in values:
                    if class_label not in baseclf_pred_prob.keys():
                        temp_predprob_list.append(-1)
                    else:
                        temp_predprob_list.append(baseclf_pred_prob[class_label])
                pred_prob_list_for_target_sample[source_keys][class_label] = temp_predprob_list

        return pred_class_list_for_target_sample, pred_prob_list_for_target_sample

       
       
       


    def eva_similarity_decisionary(self, pred_class_list_for_target_sample):
       
        similar_decisionary_flag = False
        max_prob_class_list = []

        for source_keys, values in pred_class_list_for_target_sample.items():

            values = collections.Counter(values)
            num_classifier = len(values)
            max_prob_class_list.append(values.most_common()[0])
       
       
       
       
       
       
       
       
       
        if max_prob_class_list[0][0] == max_prob_class_list[-1][0] \
                and max_prob_class_list[0][1] >= num_classifier \
                and max_prob_class_list[-1][1] >= num_classifier \
                and max_prob_class_list[0][0] != -1 \
                and max_prob_class_list[-1][0] != -1:
            similar_decisionary_flag = True

        return similar_decisionary_flag

   
   
   
   
   
   
   
   
    def eva_similarity_probability(self, pred_prob_list_for_target_sample):
       
        similar_probability_flag = False
        class_pred_list_positive = []
        class_pred_list_negative = []
        for source_keys, values in pred_prob_list_for_target_sample.items():
            if len(values) < 2:
                return False
            else:
                class_pred_list_positive.append(values.get('1'))
                class_pred_list_negative.append(values.get('0'))

        if len(class_pred_list_positive) == 2 and len(class_pred_list_negative) == 2:
            if ks_2samp(class_pred_list_negative[0],class_pred_list_negative[1]).pvalue <= 0.01 \
                    or ks_2samp(class_pred_list_positive[0],class_pred_list_positive[1]).pvalue <= 0.01:
                similar_probability_flag = False
            else:
                similar_probability_flag = True
        return similar_probability_flag

    def learning_procedure(self):

        target_domain_label = []
        target_domain_pred = []
        target_domain_joint_pred = []

        source_domain_label = []
        source_domain_pred = []

        al_labeled_window = []
        al_labeled_window_source = []

        ddm_list = []

        similar_decisionary_flag_list = []
        similar_probability_flag_list = []

        processed_sample = 0

        processed_source_sample = 0
        processed_target_sample = 0
        concept_drift_position = 0

        ms_ensemble = {}
        ms_ddmodel = {}
        ms_weight = {}

        target_supervised_ddm_model = self.creat_ddmodel_for_domain()

        for x, y in self.X_y:

            processed_sample = processed_sample + 1
           

           
            domain_id_value = 0
            domain_id_keys = ''
            for keys,values in x.items():
                domain_id_value = int(values)
                domain_id_keys = keys
                break
            x.pop(domain_id_keys)

           
            if domain_id_value not in ms_ensemble.keys():
                ms_ensemble[domain_id_value] = []
                ms_ensemble[domain_id_value].append(self.creat_model_for_domain())
                ms_ddmodel[domain_id_value] = []
                ms_ddmodel[domain_id_value].append(self.creat_ddmodel_for_domain())

                num_base_classifier = ms_ensemble[domain_id_value][-1].n_models
                ms_weight[domain_id_value] = []
                ms_weight[domain_id_value].append([i-i+1.0 for i in range(0, num_base_classifier)])

           
            temp_pred = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
            if domain_id_value == self.TARGET_DOMAIN_INDEX:
                if len(temp_pred)==0:
                    correctedpred = 1
                elif max(temp_pred.values()) >= self.drift_confidence :
                    correctedpred = 0
                elif max(temp_pred.values()) < self.drift_confidence :
                    correctedpred = 1
                in_drift, in_warning = ms_ddmodel[domain_id_value][-1].update(correctedpred)
                sigma_value = random.random()
                if sigma_value < self.AL_RANDOM_THRESHOLD:
                    if len(temp_pred) == 0:
                        correctedpred_r = 1
                    if temp_pred == y:
                        correctedpred_r = 0
                    else:
                        correctedpred_r = 1
                    in_drift_r, in_warning_r = target_supervised_ddm_model.update(correctedpred_r)
                    if in_drift_r:
                        in_drift = True
            else:
                sigma_value = random.random()
                if len(temp_pred)==0:
                    correctedpred = 1
                if temp_pred == y:
                    correctedpred = 0
                else:
                    correctedpred = 1
                in_drift, in_warning = ms_ddmodel[domain_id_value][-1].update(correctedpred)

           
            if in_drift:
                concept_drift_position = processed_sample
                print(f"MS Change detected at index {processed_sample}, in domain: {domain_id_value}")
                ms_ensemble[domain_id_value].append(self.creat_model_for_domain())
                num_base_classifier = ms_ensemble[domain_id_value][-1].n_models
                ms_weight[domain_id_value].append([i-i+1 for i in range(0, num_base_classifier)])

                for keys, values in ms_weight.items(): 
                    for j, baseem in enumerate(values): 
                        for k, baseclf in enumerate(baseem): 
                            ms_weight[keys][j][k] = 1.0
           

           
            if domain_id_value!= self.TARGET_DOMAIN_INDEX:
                processed_source_sample = processed_source_sample + 1
               
                al_labeled_flag = False
                if processed_source_sample <= self.al_ini_num:
                    al_labeled_flag = True
                elif sigma_value < self.AL_RANDOM_THRESHOLD:
                    al_labeled_flag = True
                else:
                    pred_votes = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                    if len(pred_votes) != 0:
                        max_prob_index = max(pred_votes, key=pred_votes.get)
                        max_prob = pred_votes.get(max(pred_votes, key=pred_votes.get))
                        pred_votes[max_prob_index] = 0
                        second_prob = pred_votes.get(max(pred_votes, key=pred_votes.get))
                        uncertainty_margin_value = max_prob - second_prob
                        if uncertainty_margin_value < self.AL_UNCERTAINTY_THRESHOLD:
                            al_labeled_flag = True

                if al_labeled_flag == True:
                    ms_ensemble[domain_id_value][-1].learn_one(x, y)
                    al_labeled_window_source.append(1)
                else:
                    al_labeled_window_source.append(0)

           
            else:
                if in_drift:
                    ddm_list.append(2)
                elif in_warning:
                    ddm_list.append(1)
                else:
                    ddm_list.append(0)

               
                processed_target_sample = processed_target_sample + 1
                target_domain_label.append(y)

                pred_class_list_for_target_sample, pred_prob_list_for_target_sample = self.cal_eva_similarity_problist(x, ms_ensemble)
                similar_decisionary_flag = self.eva_similarity_decisionary(pred_class_list_for_target_sample)
                similar_probability_flag = self.eva_similarity_probability(pred_prob_list_for_target_sample)
                if similar_decisionary_flag:
                    similar_decisionary_flag_list.append(1)
                else:
                    similar_decisionary_flag_list.append(0)

                if similar_probability_flag:
                    similar_probability_flag_list.append(1)
                else:
                    similar_probability_flag_list.append(0)

               
                al_labeled_flag = False
                temp_para = concept_drift_position - processed_sample + self.al_dd_num
                volatility_margin = 1 / (1 + math.exp(-max(-20, temp_para)))
                rd_threshold = (self.AL_RANDOM_THRESHOLD + volatility_margin)

                if processed_target_sample <= self.al_ini_num:
                    al_labeled_flag = True
                elif sigma_value < rd_threshold:
                    al_labeled_flag = True
                else:
                    pred_votes = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                    if len(pred_votes) != 0:
                        max_prob_index = max(pred_votes, key=pred_votes.get)
                        max_prob = pred_votes.get(max(pred_votes, key=pred_votes.get))
                        pred_votes[max_prob_index] = 0
                        second_prob = pred_votes.get(max(pred_votes, key=pred_votes.get))
                        uncertainty_margin_value = max_prob - second_prob

                        if similar_probability_flag:
                            if uncertainty_margin_value < self.AL_UNCERTAINTY_THRESHOLD * math.exp(-(self.cs_ratio - 1)):
                                al_labeled_flag = True
                        else:
                            if uncertainty_margin_value < self.AL_UNCERTAINTY_THRESHOLD:
                                al_labeled_flag = True

               
               
               
               
               
               
               

               
                predddy = self.get_votes_for_instance(x, ms_ensemble[domain_id_value][-1])
                if len(predddy) == 0:
                    predddy = y
                else:
                    predddy = max(predddy, key=predddy.get)
                target_domain_pred.append(predddy)

               
                y_joint_pred = collections.Counter()
                for keys,values in ms_ensemble.items():
                    for j, basemodel in enumerate(values):
                       
                        for k,baseclf in enumerate(basemodel):
                            temp_weight = ms_weight[keys][j][k]
                            votes = baseclf.predict_proba_one(x)
                            for key in votes:
                                votes[key] *= temp_weight
                            y_joint_pred.update(votes)
                if len(y_joint_pred) == 0:
                    y_joint_pred = y
                else:
                    total = sum(y_joint_pred.values())
                    y_joint_pred = max(y_joint_pred, key=y_joint_pred.get)

               
               
               
               
                target_domain_joint_pred.append(y_joint_pred)

                if al_labeled_flag:
                    al_labeled_window.append(1)
                    ms_ensemble[domain_id_value][-1].learn_one(x, y)
                    ms_weight = self.asymmetric_weighting(ms_ensemble, ms_weight, x, y)
                else:
                    al_labeled_window.append(0)

        print("Target domain accuracy",accuracy_score(target_domain_label, target_domain_pred))
        print("Target domain joint pred accuracy",accuracy_score(target_domain_label, target_domain_joint_pred))
        print("Target domain ensemble number",len(ms_ensemble[self.TARGET_DOMAIN_INDEX]))
        print("Acitve learning budgets:", sum(al_labeled_window) / len(al_labeled_window))
        print("Acitve learning budgets source:", sum(al_labeled_window_source) / len(al_labeled_window_source))

        return target_domain_label, \
               target_domain_joint_pred, \
               al_labeled_window, \
               ddm_list, \
               similar_decisionary_flag_list, \
               similar_probability_flag_list, \
               al_labeled_window_source
