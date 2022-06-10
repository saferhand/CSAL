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
import numpy as np

class RanVarUn:

    def __init__(self, datastream,
                 al_budgets,
                 al_ini_num,
                 al_un_thd,
                 al_un_step):

        self.X_y = datastream
        self.al_ini_num = al_ini_num
        self.al_budgets = al_budgets
        self.al_un_thd = al_un_thd
        self.al_un_step = al_un_step

    def creat_model_for_domain(self):
        model = ensemble.BaggingClassifier(
            model=(
                preprocessing.StandardScaler() |
                tree.HoeffdingTreeClassifier()
            ),
            n_models=10,
            seed=random.randint(1,1000)
        )
        return model

    def creat_ddmodel_for_domain(self):
        ddmodel = HDDM_A()
        return ddmodel

    def learning_procedure(self):

        real_label = []
        pred_label = []
        al_labeled_window = []
        ddm_list = []

        processed_sample = 0

        drift_detector = self.creat_ddmodel_for_domain()
        model = self.creat_model_for_domain()

        for x, y in self.X_y:
            processed_sample = processed_sample + 1
            al_labeled_flag = False
            sigma_value = random.random()

            real_label.append(y)
            if len(al_labeled_window) == 0:
                current_labeling_cost = 0
            else:
                current_labeling_cost = sum(al_labeled_window) / len(al_labeled_window)

            y_pred = model.predict_proba_one(x)
            if len(y_pred) <=1:
                max_prob = 1.0
            else:
                max_prob = max(y_pred.values())
            max_prob = max_prob/(np.random.normal(loc=0.0, scale=1.0, size=None) + 1)
            if (max_prob < self.al_un_thd or processed_sample < self.al_ini_num) and current_labeling_cost < self.al_budgets:
                al_labeled_flag = True
                self.al_un_thd = self.al_un_thd * (1 - self.al_un_step)
            else:
                self.al_un_thd = self.al_un_thd * (1 + self.al_un_step)

            if model.predict_one(x) == y:
                if al_labeled_flag:
                    drift_detector.update(1)
                pred_label.append(y)
            else:
                if al_labeled_flag:
                    drift_detector.update(0)
                if model.predict_one(x)!= None:
                    pred_label.append(model.predict_one(x))
                else:
                    pred_label.append(-1)

            if al_labeled_flag:
                model.learn_one(x,y)
                al_labeled_window.append(1)
            else:
                al_labeled_window.append(0)

        if drift_detector.change_detected:
            # The drift detector indicates after each sample if there is a drift in the data
            print(f'Change detected at index {processed_sample}')
            drift_detector.reset()
            model = self.creat_model_for_domain()

        print("Accuracy",accuracy_score(real_label, pred_label))
        print("Acitve learning budgets:", sum(al_labeled_window) / len(al_labeled_window))

        return real_label, \
               pred_label, \
               al_labeled_window, \
               ddm_list
