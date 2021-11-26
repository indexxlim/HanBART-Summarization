from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from soynlp.normalizer import repeat_normalize
import pandas as pd
import emoji
import re


class SequenceDataset(Dataset):
    '''
        To read Korean Corpus
    '''
    def __init__(self,data_path):
        self.df = self.read_data(data_path)
        self.df = self.df.applymap(self.clean)
        

    def __getitem__(self, index):       
        return self.df.iloc[index]
        
    def __len__(self):
        return len(self.df)
    
    
    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        elif path.endswith('json'):
            return pd.read_json(path)
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt)/json are Supported')

    def clean(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)

        return x


class SequenceGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self,batch):
        contents =  ['summarize: ' + i['news'] for i in batch]
        abstractive = [i['summary'] for i in batch]

        source_batch = self.tokenizer.batch_encode_plus(contents,
                                                        padding='max_length', #'max_length'
                                                        max_length=512,
                                                        truncation=True,
                                                        return_tensors='pt')

        target_batch = self.tokenizer.batch_encode_plus(abstractive, 
                                                        padding='max_length', 
                                                        max_length=512,
                                                        truncation=True, 
                                                        return_tensors='pt')

        target_batch.attention_mask[target_batch.attention_mask==self.pad_id] = -100                                                        
        
        return {'input_ids': source_batch.input_ids, 
                 'attention_mask': source_batch.attention_mask,
                 'labels': target_batch.input_ids, 
                 'decoder_attention_mask': target_batch.attention_mask,
                 'abstractive': abstractive
        }

class BartDataModule(pl.LightningDataModule):
    def __init__(self, train_path: str, valid_path: str, tokenizer, shuffle=True,batch_size=4, num_workers=4, sampler=None):
        super().__init__()        
        self.train_path = train_path
        self.valid_path = valid_path

        self.collate_fn = SequenceGenerator(tokenizer)
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle


    def setup(self, stage: Optional[str] = None):
        
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = SequenceDataset(self.train_path)
            self.valid_dataset = SequenceDataset(self.valid_path)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = SequenceDataset(self.valid_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size = self.batch_size,
                          shuffle=self.shuffle,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size = self.batch_size,
                          shuffle=self.shuffle,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          sampler=self.sampler)


    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          collate_fn=self.collate_fn,
                          num_workers=4,
                          sampler=self.sampler)
