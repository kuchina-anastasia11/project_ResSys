import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

import fasttext
from huggingface_hub import hf_hub_download
import random
from augmentex import WordAug

from .arguments import DataArguments
import json

class Augmenter():
    def __init__(
            self,
            augmentex_probability: float

    ):
        self.augmentex_probability = augmentex_probability
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model_aug = fasttext.load_model(model_path)
        self.actions = ["replace", "delete", "swap", "stopword", "split", "reverse", "text2emoji", "ngram"]
        self.word_aug_eng = WordAug(
                unit_prob=0.4, # Percentage of the phrase to which augmentations will be applied
                min_aug=1, # Minimum number of augmentations
                max_aug=5, # Maximum number of augmentations
                lang="eng", # supports: "rus", "eng"
                platform="pc", # supports: "pc", "mobile"
                random_seed=42,
            )
        self.word_aug_ru = WordAug(
                unit_prob=0.4, # Percentage of the phrase to which augmentations will be applied
                min_aug=1, # Minimum number of augmentations
                max_aug=5, # Maximum number of augmentations
                lang="rus", # supports: "rus", "eng"
                platform="pc", # supports: "pc", "mobile"
                random_seed=42,
            )
    
    def get_lang(self, passage):
        p = self.model_aug.predict(passage.replace("\n",''))
        if p[0][0] == "__label__eng_Latn" and p[1][0]>=0.8:
            return "eng"
        if p[0][0] == "__label__rus_Cyrl" and p[1][0]>=0.8:
            return "ru"
        return False
    def __call__(self, passages):
        if random.random() < self.augmentex_probability:
            for i, passsage in enumerate(passages):
                        lang = self.get_lang(passsage)
                        if lang == 'ru':
                            #print(passages[i])
                            try:
                                passages[i] = self.word_aug_ru.augment(text=passsage, action=random.choice(self.actions))
                            except IndexError:
                                #print("wrong str")
                                pass
                            except ValueError:
                                pass
                            #print(passages[i])
                        if lang == 'eng':
                            #print(passages[i])
                            try:
                                passages[i] = self.word_aug_eng.augment(text=passsage, action=random.choice(self.actions))
                            except IndexError:
                                pass
                            except ValueError:
                                pass
                                #print("wrong str")
        return passages



def getTrainDatasetForEmbedding(data_args: DataArguments, tokenizer: PreTrainedTokenizer):
    train_dataset = datasets.DatasetDict()
    if data_args.sampling_strategy == "dataset":
        if os.path.isdir(data_args.train_data):
                for i,file in enumerate(os.listdir(data_args.train_data)):
                    data_args.train_data +=f"/{file}"
                    train_dataset[i] = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
                    data_args.train_data = data_args.train_data.replace(f"/{file}", "")
        else:
            train_dataset[0] = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    elif data_args.sampling_strategy == "prefix":
        prefix_dict = json.load(open(data_args.prefix_config_path))
        prefix_filename_dict = {}
        for filename in prefix_dict.keys():
            prefix = str(prefix_dict[filename])
            if prefix not in prefix_filename_dict.keys():
                prefix_filename_dict[prefix] = []
            prefix_filename_dict[prefix].append(filename)
        if os.path.isdir(data_args.train_data):
                for i,prefix in enumerate(prefix_filename_dict.keys()):
                    file_names = list(set(prefix_filename_dict[prefix]) & set(os.listdir(data_args.train_data)))
                    if len(file_names):
                        train_dataset[i] = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer, name_for_one_prefix=file_names)
        else:
            train_dataset[0] = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    else:
        train_dataset[0] = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    return train_dataset

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            name_for_one_prefix = None
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            if name_for_one_prefix is not None:
                for name in name_for_one_prefix:
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, name),
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    name_column = [str(name)] * len(temp_dataset)
                    temp_dataset = temp_dataset.add_column("name", name_column)
                    train_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_datasets)
            else:
                for file in os.listdir(args.train_data):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    name_column = [str(file)] * len(temp_dataset)
                    temp_dataset = temp_dataset.add_column("name", name_column)
                    train_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')
            name_column = [args.train_data.split('/')[-1]] * len(self.dataset)
            self.dataset = self.dataset.add_column("name", name_column)


        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.augmentex_probability = args.augmentex_probability
        if args.augmentex_probability:
            self.use_augmentex = True
            self.augmenter = Augmenter(self.augmentex_probability)
        else:
            self.use_augmentex = False
        self.prefix_probability = args.prefix_probability
        self.prefix_strategy = args.prefix_strategy
        if self.prefix_probability:
            self.use_prefix = True
            self.prefix_dict = json.load(open(args.prefix_config_path))
        else:
            self.use_prefix = False
    

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        if self.use_augmentex:
            passages = self.augmenter(passages)

        if self.prefix_strategy == "sample":
                prefix = self.prefix_dict[self.dataset[item]['name']]
                if random.random() < self.prefix_probability:
                    query = prefix[0]+query
                    for i in range(len(passages)):
                        passages[i] = prefix[1]+passages[i]
        if self.prefix_strategy == "text":
                prefix = self.prefix_dict[self.dataset[item]['name']]
                if random.random() < self.prefix_probability:
                    query = prefix[0]+query
                for i in range(len(passages)):
                    if random.random() < self.prefix_probability:
                        passages[i] = prefix[1]+passages[i]


        #print(passages)
        return query, passages


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}
