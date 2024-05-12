import polars as pl
import numpy as np
import pandas as pd
vacancies = pl.read_parquet("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/hh_recsys_vacancies.pq")
vacancies = vacancies.to_pandas()
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
name_emb_df = pd.DataFrame()
vacancies = list(vacancies['description'])
vacancies_emb = []
for i in tqdm(range(len(vacancies))):
    desc = BeautifulSoup(vacancies[i]).get_text()
    print(desc)
    #vacancies_emb.append([model.encode(desc)])
    vacancies_emb.append(desc)
name_emb_df['desc'] = vacancies_emb
name_emb_df.to_json(
        "/home/jovyan/shares/SR004.nfs2/amaksimova/exp/desc_clear.jsonl", orient="records", force_ascii=False
    )
