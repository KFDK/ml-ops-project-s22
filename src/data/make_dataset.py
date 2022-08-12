# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataloader object
class Dataset():
    def __init__(self, texts, targets, tokenizer, max_len):
        self.text = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)}

def read_data():
    # Read data from raw
    fake=pd.read_csv(input_filepath+'/fake.csv')
    true=pd.read_csv(input_filepath+'/true.csv')
    fake['target']=0 # Fake
    true['target']=1 # True
    
    # Concat datasets 
    df=pd.concat([true,fake])
    
    return df

def split_data():
    # Split data to train, test, eval.
    df_train,df_test=train_test_split(df,test_size=0.4,random_state=1)
    df_test,df_eval=train_test_split(df_test,test_size=0.25,random_state=1)
    
    return df_train, df_test, df_eval

def create_dataloader(df, tokenizer, max_len, batch_size):
    # Create dataloader
    ds = Dataset(
    texts=df["text"].to_numpy(),
    targets=df['target'].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    
    return DataLoader(ds,batch_size=batch_size,num_workers=2)

def save_dataloader_as_torchdataset():
    torch.save(train_dataloader, output_filepath+'/train_dataloader.pt')
    torch.save(test_dataloader, output_filepath+'/test_dataloader.pt')
    torch.save(eval_dataloader, output_filepath+'/eval_dataloader.pt')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define tokenizer
    electra_huggingface='google/electra-small-discriminator'
    tokenizer = AutoTokenizer.from_pretrained(electra_huggingface)

    df = read_data() # read fake.csv and true.csv. return concat dataframe
    df_train, df_test, df_eval = split_data() # split data

    train_dataloader = create_dataloader(df_train, tokenizer, 200, 8) # set tokenize, max length, batch size
    test_dataloader = create_dataloader(df_test, tokenizer, 200, 8)
    eval_dataloader = create_dataloader(df_eval, tokenizer, 200, 8)

    save_dataloader_as_torchdataset() # save dataloader as dataset

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
