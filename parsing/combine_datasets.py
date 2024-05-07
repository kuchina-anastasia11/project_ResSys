import pandas as pd
import sys
from datetime import datetime

if len(sys.argv) < 2:
    print('Usage: python Combine_datasets.py <dataset_1> <dataset_2> ...')
    sys.exit(1)

all_dfs = []

for filename in sys.argv[1:]:
    df = pd.read_csv(filename)
    all_dfs.append(df)

common_df = pd.concat(all_dfs, ignore_index=True)

cur_date = datetime.now().strftime("%Y-%m-%d")

outfile = f'realtime_dataset_{cur_date}.csv'

common_df.to_csv(outfile, index=False)

print(common_df)