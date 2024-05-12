import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def calculate_similarity(row, data_skills, threshold=0.25):

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

                    str_pred = row_pred[col].values[0]
                    str_test = row_test[col].values[0]
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform([str_pred, str_test])
                    similarity = cosine_similarity(X[0], X[1])
                    similarity_features.append(similarity)
                
                features = ['compensation_from', 'employment', 'workSchedule', 'workExperience']
                for feature in features:
                    value_pred = row_pred[feature].values[0]
                    value_test = row_test[feature].values[0]
                    if feature == 'compensation_from':
                        similarity_features.append(1 if abs(value_pred - value_test) <= 0.2 * max(value_pred, value_test) else 0)
                    else:
                        similarity_features.append(1 if value_pred == value_test else 0)
                
                mean_similarity = np.mean(similarity_features)
                
                result = 0 if mean_similarity > threshold else 1
                results.append(result)
                
        return results
    
    except ValueError as e:
        print(e)
        return np.nan
