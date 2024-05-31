import numpy as np
import pandas as pd
from helpers import get_work_experience_similarity, get_compensation_similarity, get_cosine_similarity

def calculate_mrr(index_true, index_preds, vacancy, threshold=0.75):
    results = []
    
    for elem_pred in index_preds:
        row_pred = vacancy[vacancy['vacancy_id'] == elem_pred]
        if row_pred.empty:
            continue

        best = None

        for i, elem_true in enumerate(index_true):
            row_true = vacancy[vacancy['vacancy_id'] == elem_true]
            if row_true.empty:
                continue

            similarity_features = []

            column_compare = ['keySkills_str', 'name']
            for col in column_compare:
                if pd.isna(row_pred[col].values[0]) or pd.isna(row_true[col].values[0]):
                    similarity = 0
                else:
                    similarity = get_cosine_similarity(row_pred, row_true, col)
                similarity_features.append(similarity)

            similarity_features.append(get_work_experience_similarity(
                row_pred['workExperience'].values[0],
                row_true['workExperience'].values[0]
            ))

            similarity_features.append(get_compensation_similarity(
                row_pred['compensation_from'].values[0],
                row_true['compensation_from'].values[0]
            ))

            features = ['employment', 'workSchedule']
            for feature in features:
                value_pred = row_pred[feature].values[0]
                value_true = row_true[feature].values[0]
                similarity_features.append(1 if value_pred == value_true else 0)

            sim = np.mean(similarity_features)

            if sim > threshold:
                best = i + 1
                break

        if best is not None:
            results.append(1 / best)
        else:
            results.append(0)

    return np.sum(results) / len(results) if results else 0