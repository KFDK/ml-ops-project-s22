""" 
This script trains the model.

config file: train.yaml
"""

# Misc
# from select import EPOLLEXCLUSIVE
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from datetime import datetime

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# deep learning / pytorch
import torch
from torch import nn, optim
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForPreTraining,
    AdamW,
    get_scheduler,
    get_linear_schedule_with_warmup,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pdb
from transformers import ElectraModel, TrainingArguments, Trainer, AutoTokenizer
import torch
from torch import nn

from hydra import compose, initialize
from omegaconf import OmegaConf

import wandb
from google.cloud import secretmanager
import os

# import gcsfs
from google.cloud import storage
from google.cloud import secretmanager


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


def load_model(device):
    model = ElectraModel.from_pretrained(
        pretrained_model_name_or_path="google/electra-small-discriminator", num_labels=2,
        return_dict=False
    )
    model.to(device)
    return model


def my_tokenize(X):
    # Tokenize with electra. Input list of texts
    electra_huggingface = "google/electra-small-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(electra_huggingface)
    tokenizer.padding_side = "left"
    encodings = tokenizer(X, truncation=True, padding=True)

    return encodings


class ElectraClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained(
            "google/electra-small-discriminator"
        )
        self.dense1 = nn.Linear(
            self.electra.config.hidden_size, self.electra.config.hidden_size
        )
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

    def classifier(self, sequence_output):
        x = sequence_output[:, 0, :]
        x = F.gelu(self.dense1(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        sm = nn.Softmax(dim=1)
        return sm(logits)

    def forward(self, input_ids=None, attention_mask=None):
        discriminator_hidden_states = self.electra(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        pdb.set_trace()
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def save_model(model, model_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket("better-mldtu-aiplatform")
    blob = bucket.blob("models/" + model_name + ".pt")
    with blob.open("wb", ignore_flush=True) as f:
        torch.save(model, f)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# , model, train_dataset, train_dataloader, eval_dataset, eval_dataloader, EPOCH
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # wandb.init(project="mlops_wandb_project", entity="gahk_mlops")with wandb.init(config=config):
        print(config.keys())
        learning_rate = config.learning_rate
        optimizer = get_optimizer(learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        best_accuracy = 0
        for epoch in tqdm(range(EPOCHS)):
            print("starting train_epoch")
            train_acc, train_loss = train_epoch(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(train_dataset),
            )

            val_acc, val_loss = eval_model(
                model, eval_dataloader, loss_fn, device, len(eval_dataset)
            )

            wandb.log(
                {
                    "Training_loss": train_loss,
                    "Validation_loss": val_loss,
                    "Training_accuracy": train_acc,
                    "Validation_accuracy": val_acc,
                }
            )

            if val_acc > best_accuracy:
                save_model(model, "model_test")
                best_accuracy = val_acc


def get_secret():
    client = secretmanager.SecretManagerServiceClient()
    secret_path = "projects/822364161295/secrets/wandb_api_key"
    secret = client.access_secret_version(secret_path)
    return secret.payload.data.decode("UTF-8")


def access_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


def init_wandb(key):
    wandb.login(key=key)
    wandb.init(project="mlops_wandb_project", entity="gahk_mlops")
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_API_KEY"] = key
    wandb_agent = "wandb agent " + "gahk_mlops" + "/" + "mlops_wandb_project"
    os.system(wandb_agent)


def freeze_electra():
    for p in model.electra.parameters():
        p.requires_grad = False


def get_optimizer(learning_rate):
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        correct_bias=False,
        no_deprecation_warning=True,
    )
    return optimizer


# Init hyperparameters
config_path = "./configs/"
configs = OmegaConf.load(config_path + "train.yaml")
configs_secret = OmegaConf.load(config_path + "secret.yaml")

# Hyperparameters extracted
# learning_rate = configs.hyperparameters.learning_rate
EPOCHS = configs.hyperparameters.epochs
batch = configs.hyperparameters.batch_size
seed = configs.hyperparameters.seed
model_name = configs.hyperparameters.model_name


if __name__ == "__main__":
    
    sweep_configuration = OmegaConf.load(config_path + "sweep.yaml")
    sweep_configuration = OmegaConf.to_container(sweep_configuration)

    # sweep_config={
    # "name": "my_test_sweep",
    # "method": "bayes", 
    # "metric": {"name": "Validation_accuracy", "goal": "maximize"}, 
    # "parameters": {
    #     "learning_rate": {
    #         "values": [0.0001, 0.001]
    #         }
    #     }
    # }

    api_key = configs_secret.hyperparameters.wandb_api_key
    wandb.login(key=api_key)

    # Set seed for reproducibility.
    # set_seed(seed)
    # torch.manual_seed(seed)
    dt = str(datetime.now())[:16]

    data_output_filepath = "data/processed/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ElectraClassifier()
    model = model.to(device)
    # wandb.watch(model, log_freq=100)
    freeze_electra()
    train_dataset = torch.load(data_output_filepath + "train_dataset.pt")
    eval_dataset = torch.load(data_output_filepath + "eval_dataset.pt")
    train_dataloader = DataLoader(train_dataset, batch_size=batch)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch)
    total_steps = len(train_dataloader) * EPOCHS
    loss_fn = nn.CrossEntropyLoss().to(device)
    print(len(train_dataloader))
    # train(model, train_dataset, train_dataloader, eval_dataset, eval_dataloader, EPOCHS)

    sweep_id = wandb.sweep(sweep_configuration,project="mlops_wandb_project")
    wandb.agent(sweep_id, function=train,count=3)
    print("done!")