import pandas as pd
net = pd.read_json('/home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/close_names_descriprion.json')
neg = pd.read_json("/home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/different_names_descriprion.json")
pos = pd.read_json('/home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/cool_tune_small.json')
dataset = []
for i in range(len(pos)):
    data={}
    data[1] = [list(pos)['query'][i], list(pos)['pos'][i][0]]
    data[2] = [list(net)['query'][i], list(net)['pos'][i][0]]
    data[3] = [list(neg)['query'][i], list(neg)['pos'][i][0]]
    
from flag_models import FlagModel
model_pth = '/home/jovyan/shares/SR004.nfs2/amaksimova/exp/10/finetune/out_cool_data_bibert'
model = FlagModel(model_pth, 
                 # query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)
num_true = 0
for data in dataset:
    
    embeddings_1 = model.encode(data[1][0])
    embeddings_2 = model.encode(data[1][0])
    similarity1 = embeddings_1 @ embeddings_2.T
    
    embeddings_1 = model.encode(data[2][0])
    embeddings_2 = model.encode(data[2][0])
    similarity2 = embeddings_1 @ embeddings_2.T
    
    embeddings_1 = model.encode(data[3][0])
    embeddings_2 = model.encode(data[3][0])
    similarity3 = embeddings_1 @ embeddings_2.T
    
    if similarity1>similarity2 and similarity2>similarity3:
        num_true+=1
print(num_true/50)
