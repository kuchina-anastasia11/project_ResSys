#считаем эмбединги
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
device = "cuda"
model = SentenceTransformer('sentence-transformers/LaBSE').to(device)
name_emb_df = pd.DataFrame()
names_ds = pd.read_json("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/names.jsonl")
clean_names = list(names_ds['clean_names'])
slash_names = list(names_ds['slash_names'])
parenthesis = list(names_ds['parenthesis'])
name = list(names_ds['name'])
clean_names_emb = []
slash_names_emb = []
parenthesis_emb = []
name_emb = []
for i in tqdm(range(len(names_ds))):
    clean_names_emb.append([model.encode(clean_names[i])])
    name_emb.append([model.encode(name[i])])
    slash_names_emb_t = []
    parenthesis_emb_t = []
    for j in slash_names[i]:
        slash_names_emb_t.append([model.encode(j)])
    slash_names_emb.append(slash_names_emb_t)
    for j in parenthesis[i]:
        parenthesis_emb_t.append([model.encode(j)])
    parenthesis_emb.append(parenthesis_emb_t)
name_emb_df['clean_names'] = clean_names_emb
name_emb_df['slash_names'] = slash_names_emb
name_emb_df['parenthesis'] = parenthesis_emb
name_emb_df['name'] = name_emb
name_emb_df.to_json(
        "/home/jovyan/shares/SR004.nfs2/amaksimova/exp/name_emb.jsonl", orient="records", force_ascii=False
    )