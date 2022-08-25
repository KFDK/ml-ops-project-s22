""" 
This script converts raw data into tokenized text using google electra. 
It returns and saves a torch dataset object to ./data/processed/
"""

# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from transformers import AutoTokenizer
# from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# Configs
from omegaconf import OmegaConf

config_path = "./configs/"
configs = OmegaConf.load(config_path + "makedata.yaml")

# Hyperparameters extracted
input_filepath = configs.hyperparameters.input_filepath
output_filepath = configs.hyperparameters.output_filepath
small_test = configs.hyperparameters.small_test


# Torch Dataset Object
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def __select__(self, idx_from, idx_to):
        items = {key: val[idx_from:idx_to] for key, val in self.encodings.items()}
        return TorchDataset(items, self.labels[idx_from:idx_to])


def read_data(input_filepath):
    """Read data from raw. returns as pandas dataframe"""
    if small_test:
        fake = pd.read_csv(input_filepath + "/Fake.csv")[:200]
        true = pd.read_csv(input_filepath + "/True.csv")[:200]
        print("small test set enabled! size:" + str(len(fake) * 2))
        fake["target"] = 0  # Fake
        true["target"] = 1  # True
        df = pd.concat([true, fake])
    else:
        fake = pd.read_csv(input_filepath + "/Fake.csv")
        true = pd.read_csv(input_filepath + "/True.csv")
        fake["target"] = 0  # Fake
        true["target"] = 1  # True
        df = pd.concat([true, fake])
    return df

def read_data_imdb(input_filepath):
    """Read data from raw. returns as pandas dataframe"""
    if small_test:
        df = pd.read_csv(input_filepath + "/imdb_data.csv")[:2000]
        print("small test set enabled! size:" + str(len(df)))
    else:
        df = pd.read_csv(input_filepath + "/imdb_data.csv")
    return df


def split_data(df):
    """split pandas dataframe"""
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=1)
    df_test, df_eval = train_test_split(df_test, test_size=0.25, random_state=1)
    return df_train, df_test, df_eval


def my_tokenize(X):
    """Tokenize with electra. Input list of texts"""
    electra_huggingface = "google/electra-small-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(electra_huggingface)
    tokenizer.padding_side = "left"
    encodings = tokenizer(X, truncation=True, padding=True)

    return encodings  # tuple with input_ids and masks


def convert_to_torchdataset(
    train_encodings, test_encodings, eval_encodings, y_train, y_test, y_eval
    ):
    """Convert to PyTorch Datasets class"""
    train_set = TorchDataset(train_encodings, y_train.to_list())
    test_set = TorchDataset(test_encodings, y_test.to_list())
    eval_set = TorchDataset(eval_encodings, y_eval.to_list())
    return train_set, test_set, eval_set


def save_dataloader_as_torchdataset(output_filepath, train_set, test_set, eval_set):
    """saves data"""
    torch.save(train_set, output_filepath + "/train_dataset.pt")
    torch.save(test_set, output_filepath + "/test_dataset.pt")
    torch.save(eval_set, output_filepath + "/eval_dataset.pt")


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    df = read_data_imdb(input_filepath)
    df_train, df_test, df_eval = split_data(df)

    train_encodings = my_tokenize(df_train["text"].to_list())
    test_encodings = my_tokenize(df_test["text"].to_list())
    eval_encodings = my_tokenize(df_eval["text"].to_list())

    y_train = df_train["target"]
    y_test = df_test["target"]
    y_eval = df_eval["target"]

    train_set, test_set, eval_set = convert_to_torchdataset(
        train_encodings, test_encodings, eval_encodings, y_train, y_test, y_eval
    )
    save_dataloader_as_torchdataset(output_filepath, train_set, test_set, eval_set)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(input_filepath, output_filepath)
