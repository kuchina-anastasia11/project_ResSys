import pandas as pd
df = pd.read_json("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/desc_emb.jsonl")
import numpy as np
a = np.array(list(df['desc']))
a = np.squeeze(a)
from numpy import dot
from numpy.linalg import norm
def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))
from tqdm import tqdm
import heapq
ans = []
for i in tqdm(range(len(a))):
    s = []
    for j in range(len(a)):
        if i!=j:
            s.append(cos_sim(a[i], a[j]))
    smallest_indices = heapq.nsmallest(
    50, range(len(s)), key=s.__getitem__)
    ans.append(smallest_indices)
dd = pd.DataFrame()
dd['near'] = ans
dd.to_json(
        "/home/jovyan/shares/SR004.nfs2/amaksimova/exp/desc_near.jsonl", orient="records", force_ascii=False
    )    
    