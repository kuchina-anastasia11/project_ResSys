import pandas as pd
df = pd.read_json("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/vacancy_for_clustering.jsonl")
print("gogogog")
w = []
for i in list(df['workExperience']):
    if i=='noExperience':
        w.append(0)
    if i=="between3And6":
        w.append(1)
    if i=="between1And3":
        w.append(2)
    if i=="moreThan6":
        w.append(3)
df['workExperience'] = w
w = []
for i in list(df['workSchedule']):
    if i=='fullDay':
        w.append(0)
    if i=="flyInFlyOut":
        w.append(1)
    if i=="flexible":
        w.append(2)
    if i=="shift":
        w.append(3)
    if i=="remote":
        w.append(4)
df['workSchedule'] = w
w = []
for i in list(df['employment']):
    if i=='full':
        w.append(4)
    if i=="part":
        w.append(3)
    if i=="project":
        w.append(2)
    if i=="probation":
        w.append(1)
    if i=="volunteer":
        w.append(0)
df['employment'] = w
df = df.drop(columns=['vacancy_id'])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['compensation.currencyCode'] = le.fit_transform(df["compensation.currencyCode"].values)
le = LabelEncoder()
df['area.id'] = le.fit_transform(df["area.id"].values)
le = LabelEncoder()
df['area.regionId'] = le.fit_transform(df["area.regionId"].values)
le = LabelEncoder()
df['company.id'] = le.fit_transform(df["company.id"].values)
df1 = pd.DataFrame()

df1[[i for i in range(768)]] = pd.DataFrame(df.name.values.tolist(), index= df.index)
df1[[i for i in range(768, 768*2)]] = pd.DataFrame(df.desc.values.tolist(), index= df.index)
df = df.drop(columns=['name','desc'])
df = pd.concat([df1, df], axis=1)
print("df ready")
import numpy as np
val = df.values
from sklearn.cluster import KMeans
import numpy as np
import torch
x  = torch.tensor(val)
b = np.where(np.isnan(x), 0, x)
print("run kmeans")
kmeans = KMeans(n_clusters=500, random_state=0, n_init="auto").fit(b)
data = kmeans.labels_
import json

with open('train.txt', 'w') as filehandle:
    json.dump(data.tolist(), filehandle)
"""with open("train.txt", "w") as txt_file:
    for line in data:
        txt_file.write(" ".join(line) + "\n") """