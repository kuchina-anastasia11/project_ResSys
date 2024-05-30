import numpy as np
import random
import pandas as pd
from utils import FlagModel, name_processing, experience_code, find_nearest_emb

class initial_recommendation():
    def __init__(self, data_vacancy: pd.DataFrame):
        self.data_vacancy = data_vacancy
        self.data_vacancy['name'] = [name_processing(i) for i in list(data_vacancy['name'])]
        self.data_vacancy['workExperience'] = [experience_code(row) for row in list(self.data_vacancy['workExperience'])]
        self.data_vacancy.compensation_from = pd.to_numeric(self.data_vacancy.compensation_from, errors='coerce').fillna(100000).astype(np.int64)
        self.model = FlagModel("models/multilingual-e5-large")
        self.names_d = {}
        names = list(self.data_vacancy['name'])
        vacancy_id = list(self.data_vacancy['vacancy_id'])
        for i in range(len(data_vacancy)):
            if names[i] not in self.names_d.keys():
                self.names_d[names[i]] = []
            self.names_d[names[i]].append(vacancy_id[i])
        
        self.nlp_cluster_size = 15000
        self.salary_coeff = 0.8
    def recomend(self, name: str, compensation_from: int, area_id: str, workExperience: str):
        data_vacancy = self.data_vacancy
        data_vacancy = data_vacancy.loc[data_vacancy['area_id'] == area_id]
        
        if len(data_vacancy)>self.nlp_cluster_size:
            data_vacancy = data_vacancy.loc[data_vacancy['workExperience'] <= experience_code(workExperience)]
        if len(data_vacancy)>self.nlp_cluster_size: 
            data_vacancy = data_vacancy.loc[data_vacancy['compensation_from'] >= compensation_from * self.salary_coeff]

        names = list(data_vacancy['name'])
        res = []
        if len(names)<5:
            names = list(self.data_vacancy['name'])
        if len(names)>self.nlp_cluster_size:
            random.shuffle(names)
            names = names[:self.nlp_cluster_size]
        
        elif len(names)<50 and len(names)>=5:
            for i in names:
                res.extend(self.names_d[i])
            return res
        
        vacancy_names = find_nearest_emb(name, names, self.model)
        for i in vacancy_names:
            res.extend(self.names_d[i])
        return res
        
    
