import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def get_cosine_similarity(vacancy_cool, comparison_vacancy, column):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([vacancy_cool[column].values[0],
                                  comparison_vacancy[column].values[0]])
    return cosine_similarity(X[0], X[1])[0][0]
