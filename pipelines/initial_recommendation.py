import numpy as np
import random
import pandas as pd
from utils import FlagModel, name_processing, experience_code, find_nearest_emb
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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

    def recomend(self, name: str, compensation_from: int, area_id: str, workExperience: str, keySkills: list): 
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
            #return res
        
        vacancy_names = find_nearest_emb(name, names, self.model)
        for i in vacancy_names:
            res.extend(self.names_d[i])

        if len(keySkills) == 0:
            return res[:5]
        
        keySkills_str = [skill.lower() for skill in keySkills]
        keySkills_str = ' '.join(keySkills_str)
        
        row = pd.DataFrame({
            'clean_name': [name],
            'compensation_from': [compensation_from],
            'area_regionId': [area_id],
            'workExperience': [workExperience],
            'keySkills': [keySkills],
            'keySkills_str': [keySkills_str],
        })
                
        need_vac_df = data_vacancy[data_vacancy['vacancy_id'].isin(res)].copy()
        need_vac_df.dropna(subset=['keySkills_str'], inplace=True)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(need_vac_df['keySkills_str'])
        
        if len(need_vac_df) > 30:
            n_neighbors = 30
        else:
            n_neighbors = len(need_vac_df)

        knn_model_str = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        knn_model_str.fit(X)

        combined_df = pd.DataFrame()

        try:
            query = vectorizer.transform(row['keySkills_str'])
        except Exception:
            return res[:5]
            
        distances, indices = knn_model_str.kneighbors(query, n_neighbors=n_neighbors)
        vacancy_neighbors = need_vac_df.iloc[indices[0]].copy()
        vacancy_neighbors['distance_str'] = distances[0]
        combined_df = pd.concat([combined_df, vacancy_neighbors])            
        cat_columns = ['workExperience']

        scaler = MinMaxScaler()
        combined_df[['compensation_from']] = scaler.fit_transform(combined_df[['compensation_from']])
        row[['compensation_from']] = scaler.transform(row[['compensation_from']])

        knn_model_cat = NearestNeighbors(n_neighbors=len(combined_df), metric='euclidean')

        try:
            knn_model_cat.fit(combined_df[cat_columns])
        except Exception as e:
            print(e)
            return res[:5]

        query = combined_df[cat_columns].values
        distances, indices = knn_model_cat.kneighbors(query, n_neighbors=len(combined_df))
        combined_df['distance_cat'] = distances.mean(axis=1)

        scaler_cat = MinMaxScaler()
        combined_df[['distance_cat']] = scaler_cat.fit_transform(combined_df[['distance_cat']])
        scaler_str = MinMaxScaler()
        combined_df[['distance_str']] = scaler_str.fit_transform(combined_df[['distance_str']])
        combined_df['average_distance'] = combined_df[['distance_str', 'distance_cat']].mean(axis=1)
        combined_df.sort_values(by='average_distance', inplace=True)

        user_neighbors = combined_df.head(5)['vacancy_id'].tolist()

        return user_neighbors
