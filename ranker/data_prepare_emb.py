
import json
users = json.load(open("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/ranker/data/users.json"))
vacancy = json.load(open("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/ranker/data/vacancy.json"))
import pandas as pd
rs=pd.read_csv("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/ranker/data/relavence_scores.csv")
rs = rs.reset_index()  # make sure indexes pair with number of rows

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/LaBSE')
def cosine(sentences):
    embeddings = model.encode(sentences,normalize_embeddings=True)
    return embeddings[0] @ embeddings[1].T
mean_names = []
mean_desc = []
from tqdm import tqdm
for index, row in tqdm(rs.iterrows()):
    if index%10000:
        print(index)
    vacancy_id = row['vacancy_id']
    vacancy_row = vacancy[vacancy_id]
    user_row = users[row['user_id']]
    names = []
    descs = []
    ind = -1
    if vacancy_id in user_row['vacancy_id']:
        ind = user_row['vacancy_id'].index(vacancy_id)

    for i in range(len(user_row['action_type'])):
        if i!=ind and user_row['action_type'][i]!=2:
            names.append(vacancy[user_row['vacancy_id'][i]]['name'])
            descs.append(vacancy[user_row['vacancy_id'][i]]['description'])

    desc_v = vacancy[vacancy_id]['description']
    name_v = vacancy[vacancy_id]['name']
    s_n = []
    s_v = []
    for i in range(len(names)):
        s_n.append(cosine([name_v, names[i]]))
        s_v.append(cosine([desc_v, descs[i]]))
    if len(s_n):
        mean_names.append(sum(s_n)/len(s_n))
        mean_desc.append(sum(s_v)/len(s_v))
    else:
        mean_names.append(0)
        mean_desc.append(0)

rs['mean_names'] = mean_names
rs['mean_desc'] = mean_desc
rs.to_csv("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/ranker/data/df.csv")