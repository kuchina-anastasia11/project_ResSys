import joblib
import pandas as pd
import pickle
from initial_recommendation import initial_recommendation
import numpy as np
from utils import (
    name_processing,
    desc_processing,
    action_type_processor,
    key_skills_coverage,
)
from sentence_transformers import SentenceTransformer


class predictor():
    def __init__(
        self,
        data_vacancy: pd.DataFrame,
    ):
        self.ranker = joblib.load("models/LGBMRanker.pkl")
        self.salary_predictor = pickle.load(open("models/salary_pred.pkl", "rb"))
        self.encoder = SentenceTransformer("sentence-transformers/LaBSE")
        self.data_vacancy = data_vacancy
        self.initial_recommendator = initial_recommendation(self.data_vacancy)

        self.data_dict = data_vacancy
        self.data_dict["vacancy_id"] = list(data_vacancy["vacancy_id"])
        self.data_dict = self.data_dict.set_index("vacancy_id")

    def cos_sim(self, sentence1, sentence2):
        embeddings = self.encoder.encode(
            [sentence1, sentence2], normalize_embeddings=True
        )
        return embeddings[0] @ embeddings[1].T

    def get_names_desc_sim(self, users_names, vacancy_name, users_desc, vacancy_desc):
        nnames = []
        ddesc = []
        for n in users_names:
            nnames.append(self.cos_sim(name_processing(n), vacancy_name))
        for v in users_desc:
            ddesc.append(self.cos_sim(desc_processing(v), vacancy_desc))
        return sum(nnames) / len(nnames), sum(ddesc) / len(ddesc)

    def recommend(
        self,
        name: str,
        area_id: str,
        workExperience: str,
        action_type: list,
        vacancy_id_list: list,
        keySkills: list,
    ):
        users_vacancy_salary_to = []
        users_vacancy_salary_from = []
        users_vacancy_name = []
        users_vacancy_desc = []
        users_vacancy_keyskills = []
        for vacancy_id in vacancy_id_list:
            print(vacancy_id)
            vacancy = self.data_dict.loc[vacancy_id]

            users_vacancy_salary_to.append(vacancy["compensation_to"])
            users_vacancy_salary_from.append(vacancy["compensation_from"])
            users_vacancy_name.append(name_processing(vacancy["name"]))
            users_vacancy_desc.append(desc_processing(vacancy["description"]))
            users_vacancy_keyskills.extend(vacancy["keySkills"])
        users_vacancy_keyskills = set(users_vacancy_keyskills)
        pred_salary = self.salary_predictor.predict(
            action_type_processor(action_type, users_vacancy_salary_to)
        )[0]

        pred_id = self.initial_recommendator.recomend(
            name, pred_salary, area_id, workExperience, keySkills
        )
        if len(pred_id) > 10:
            pred_id = pred_id[:10]

        vacancy_names = []
        vacancy_desc = []
        vacancy_salary_from = []
        vacancy_salary_to = []
        vacancy_keyskills = []

        for vacancy_id in pred_id:

            vacancy = self.data_dict.loc[vacancy_id]
            n, d = self.get_names_desc_sim(
                users_vacancy_name,
                vacancy["name"],
                users_vacancy_desc,
                vacancy["description"],
            )
            vacancy_names.append(n)

            vacancy_desc.append(d)

            vacancy_keyskills.append(
                key_skills_coverage(users_vacancy_keyskills, set(vacancy["keySkills"]))
            )

            vacancy_salary_to.append(vacancy["compensation_to"])
            vacancy_salary_from.append(vacancy["compensation_from"])

        ranker_data = pd.DataFrame()
        ranker_data["mean_names"] = vacancy_names
        ranker_data["mean_desc"] = vacancy_desc
        ranker_data["user_salary_from"] = vacancy_salary_from
        ranker_data["user_salary_to"] = vacancy_salary_to
        ranker_data["vacancy_salary_from"] = np.mean(users_vacancy_salary_from)
        ranker_data["vacancy_salary_to"] = np.mean(users_vacancy_salary_to)
        ranker_data["intersection_keyskill"] = vacancy_keyskills
        ranker_data["vacancy_id"] = pred_id
        ranker_data = ranker_data[ranker_data["mean_names"] > 0]
        ranker_data["vacancy_salary_from"] = ranker_data["vacancy_salary_from"].fillna(
            (ranker_data["vacancy_salary_from"].mean())
        )
        ranker_data["vacancy_salary_to"] = ranker_data["vacancy_salary_to"].fillna(
            (ranker_data["vacancy_salary_to"].mean())
        )
        pred_id = list(ranker_data["vacancy_id"])
        pred = self.ranker.predict(ranker_data.drop(columns="vacancy_id"))
        ans = []
        for i in range(len(pred)):
            ans.append([pred[i], pred_id[i]])
        ans = sorted(ans)
        ans = [i[1] for i in ans]
        return ans
