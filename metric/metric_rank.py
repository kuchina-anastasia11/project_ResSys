import numpy as np
import pandas as pd
from helpers import get_work_experience_similarity, get_compensation_similarity, get_cosine_similarity


def precision_at_k(y_true, y_pred, k=10):
    
    y_pred_k = y_pred[:k]
    true_positives = set(y_pred_k).intersection(set(y_true))
    precision = len(true_positives) / min(k, len(y_true))
    
    return precision


def recall_at_k(y_true, y_pred, k=10):

    y_pred_k = y_pred[:k]
    true_positives = set(y_pred_k).intersection(set(y_true))
    recall = len(true_positives) / len(y_true)
    
    return recall


def calculate_similarity(row, data_skills, threshold=0.75):
    try:
        test_list = row['vacancy_id_true']
        preds_list = row['vacancy_id_preds']
        results = []

        for elem_test in test_list:
            for elem_pred in preds_list:
                row_pred = data_skills[data_skills['vacancy_id'] == elem_pred]
                row_test = data_skills[data_skills['vacancy_id'] == elem_test]
                if row_pred.empty or row_test.empty:
                    results.append(0)
                    continue

                similarity_features = []

                column_compare = ['keySkills_str', 'name']
                for col in column_compare:
                    if pd.isna(row_pred[col].values[0]) or pd.isna(row_test[col].values[0]):
                        similarity = 0
                    else:
                        similarity = get_cosine_similarity(row_pred, row_test, col)
                    similarity_features.append(similarity)

                similarity_features.append(get_work_experience_similarity(
                    row_pred['workExperience'].values[0],
                    row_test['workExperience'].values[0]
                ))

                similarity_features.append(get_compensation_similarity(
                    row_pred['compensation_from'].values[0],
                    row_test['compensation_from'].values[0]
                ))

                features = ['employment', 'workSchedule']
                for feature in features:
                    value_pred = row_pred[feature].values[0]
                    value_test = row_test[feature].values[0]
                    similarity_features.append(1 if value_pred == value_test else 0)

                mean_similarity = np.mean(similarity_features)

                result = 1 if 1 - mean_similarity > threshold else 0
                results.append(result)

        return np.mean(results)

    except ValueError:
        return None