import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_calculate(row, column_compare, data_skills, max_indices=3):
    try:
        test_list = row['vacancy_id_true']
        preds_list = row['vacancy_id_preds']
        cos_dst = 0

        all_lists = [test_list, preds_list]

        for index, each_list in enumerate(all_lists):
            i = 0
            while i < len(each_list):
                row = data_skills[data_skills['vacancy_id'] == each_list[i]]
                str_elem = row[column_compare]
                if len(str_elem) == 0:
                    del each_list[i]
                else:
                    i += 1

            if len(each_list) > max_indices:
                test_indices = np.random.choice(len(each_list), max_indices, replace=False)
                all_lists[index] = [each_list[i] for i in test_indices]

        test_list = all_lists[0]
        preds_list = all_lists[1]

        cnt = len(test_list) * len(preds_list)

        for elem_test in test_list:
            for elem_pred in preds_list:
                row_pred = data_skills[data_skills['vacancy_id'] == elem_pred]
                str_pred = row_pred[column_compare]
                row_test = data_skills[data_skills['vacancy_id'] == elem_test]
                str_test = row_test[column_compare]
                if len(str_test) == 0 or len(str_pred) == 0:
                    cnt -= 1
                    if cnt == 0:
                        return np.nan
                    else:
                        continue
                str_pred = str_pred.values[0]
                str_test = str_test.values[0]
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform([str_pred, str_test])
                similarity = cosine_similarity(X[0], X[1])
                cos_dst += similarity

    except ValueError:
        return np.nan
    
    if cnt == 0:
        return np.nan
    
    return cos_dst / cnt
