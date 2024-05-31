import pandas as pd
from helpers import get_work_experience_similarity, get_compensation_similarity, get_cosine_similarity


def ranking_test(vacancy_cool_id, comparison_list, vacancy):
    vacancy_cool = vacancy[vacancy['vacancy_id'] == vacancy_cool_id]

    similarity_dict = {}

    weights = {
        'keySkills_str': 0.25,
        'name': 0.25,
        'workExperience': 0.15,
        'compensation_from': 0.15,
        'employment': 0.1,
        'workSchedule': 0.1
    }

    for comp_vacancy_id in comparison_list:
        comp_vacancy = vacancy[vacancy['vacancy_id'] == comp_vacancy_id]

        if comp_vacancy.empty:
            continue

        similarity_features = []

        for col in ['keySkills_str', 'name']:
            if pd.isna(vacancy_cool[col].values[0]) or pd.isna(comp_vacancy[col].values[0]):
                similarity = 0
            else:
                similarity = get_cosine_similarity(vacancy_cool, comp_vacancy, col)
            similarity_features.append(weights[col] * similarity)

        if pd.isna(vacancy_cool['workExperience'].values[0]) or \
            pd.isna(comp_vacancy['workExperience'].values[0]):
            similarity = 0
        else:
            similarity = get_work_experience_similarity(
                vacancy_cool['workExperience'].values[0],
                comp_vacancy['workExperience'].values[0]
            )
        similarity_features.append(weights['workExperience'] * similarity)

        if pd.isna(vacancy_cool['compensation_from'].values[0]) or \
            pd.isna(comp_vacancy['compensation_from'].values[0]):
            similarity = 0
        else:
            similarity = get_compensation_similarity(
                vacancy_cool['compensation_from'].values[0],
                comp_vacancy['compensation_from'].values[0]
            )
        similarity_features.append(weights['compensation_from'] * similarity)

        for feature in ['employment', 'workSchedule']:
            if pd.isna(vacancy_cool[feature].values[0]) or \
                pd.isna(comp_vacancy[feature].values[0]):
                similarity = 0
            else:
                value_cool = vacancy_cool[feature].values[0]
                value_comp = comp_vacancy[feature].values[0]
                similarity = 1 if value_cool == value_comp else 0
            similarity_features.append(weights[feature] * similarity)

        weighted_similarity = sum(similarity_features)

        similarity_dict[comp_vacancy_id] = weighted_similarity

    return similarity_dict
