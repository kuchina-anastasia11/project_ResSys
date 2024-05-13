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


work_experience_map = {
    'noExperience': 0,
    'between1And3': 1,
    'between3And6': 2,
    'moreThan6': 3
}


def get_work_experience_similarity(val_1, val_2):
    idx1 = work_experience_map[val_1]
    idx2 = work_experience_map[val_2]

    diff = abs(idx1 - idx2)
    if diff == 0:
        return 1
    elif diff == 1:
        return 2 / 3
    elif diff == 2:
        return 1 / 3
    else:
        return 0
    
    
def get_compensation_similarity(val_1, val_2):
    if np.isnan(val_1) or np.isnan(val_2):
        return 0
    if abs(val_1 - val_2) > 0.5 * max(val_1, val_2):
        return 0
    else:
        return 1 - abs(val_1 - val_2) / (0.5 * max(val_1, val_2))


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
                    str_pred = row_pred[col].values[0]
                    str_test = row_test[col].values[0]
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform([str_pred, str_test])
                    similarity = cosine_similarity(X[0], X[1])
                    similarity_features.append(1 - similarity[0][0])
                
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

                result = 1 if mean_similarity > threshold else 0
                results.append(result)

        return np.mean(results)
            
    except ValueError as e:
        print(e)
