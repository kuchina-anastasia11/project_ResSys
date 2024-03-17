import polars as pl
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import implicit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
RANDOM_STATE = 42
N_PREDICTIONS = 100
vacancies = pl.read_parquet("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/hh_recsys_vacancies.pq")
vacancies = vacancies.to_pandas()
#считаем эмбединги
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
device = "cuda"
model = SentenceTransformer('sentence-transformers/LaBSE').to(device)
name_emb_df = pd.DataFrame()
vacancies = list(vacancies['description'])
vacancies_emb = []
for i in tqdm(range(len(vacancies))):
    desc = BeautifulSoup(vacancies[i]).get_text()
    
    #vacancies_emb.append([model.encode(desc)])
    vacancies_emb.append(desc)
name_emb_df['desc'] = vacancies_emb
name_emb_df.to_json(
        "/home/jovyan/shares/SR004.nfs2/amaksimova/exp/desc_clear.jsonl", orient="records", force_ascii=False
    )