""" 
This script trains the model.

config file: train.yaml
"""
import pdb

# Misc
# from select import EPOLLEXCLUSIVE
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

# deep learning / pytorch
import torch
from torch import nn, optim
from transformers import (
    set_seed,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers import (
    AdamW,
    GPT2Config,
    GPT2ForSequenceClassification,
    set_seed,
    get_scheduler,
    WEIGHTS_NAME,
    CONFIG_NAME,
)

import torch.nn.functional as F
from torch.utils.data import DataLoader

# import pdb
from transformers import ElectraModel, AutoTokenizer
import torch
from torch import nn

from omegaconf import OmegaConf

import wandb
from google.cloud import secretmanager
import os

# import gcsfs
from google.cloud import storage
from google.cloud import secretmanager

# import model
from model import ElectraClassifier

# profiling
from torch.profiler import profile, record_function, ProfilerActivity


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
        pretrained_model_name_or_path="google/electra-small-discriminator",
        num_labels=2,
        return_dict=False,
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


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
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


def save_model(model, time):
    folder = "models/"
    save_path = folder + time
    os.mkdir(save_path)

    # save locally
    torch.save(model.state_dict(), os.path.join(save_path, WEIGHTS_NAME))

    # save in cloud
    bucket_name = "better-mldtu-aiplatform"
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(os.path.join(save_path, WEIGHTS_NAME))
    blob.upload_from_filename(os.path.join(save_path, WEIGHTS_NAME))


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


def train(config=None):
    with wandb.init(config=config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ElectraClassifier()
        model = model.to(device)

        # freze electra
        for p in model.electra.parameters():
            p.requires_grad = False

        loss_fn = nn.CrossEntropyLoss().to(device)

        config = wandb.config

        if configs_train.hyperparameters.sweeping:
            batch = config.batch_size
            learning_rate = config.learning_rate
            optimizer_algorithm = config.optimizer_algorithm
        else:
            batch = configs_train.hyperparameters.batch_size
            learning_rate = configs_train.hyperparameters.learning_rate
            optimizer_algorithm = configs_train.hyperparameters.optimizer_algorithm

        train_dataloader = DataLoader(train_dataset, batch_size=batch)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch)
        optimizer = get_optimizer(learning_rate, optimizer_algorithm, model)
        total_steps = len(train_dataloader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        best_accuracy = 0
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(time)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for epoch in tqdm(range(EPOCHS)):
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
                        best_accuracy = val_acc
                        if not configs_train.hyperparameters.sweeping:
                            save_model(model, time)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


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


def get_optimizer(learning_rate, optimizer_algorithm, model):
    if optimizer_algorithm == "adam":
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            correct_bias=False,
            no_deprecation_warning=True,
        )
    if optimizer_algorithm == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return optimizer


if __name__ == "__main__":

    # Init hyperparameters
    config_path = "./configs/"
    configs_train = OmegaConf.load(config_path + "train.yaml")
    configs_secret = OmegaConf.load(config_path + "secret.yaml")

    # Hyperparameters extracted
    EPOCHS = configs_train.hyperparameters.epochs
    seed = configs_train.hyperparameters.seed
    model_name = configs_train.hyperparameters.model_name

    sweep_configuration = OmegaConf.load(config_path + "sweep.yaml")
    sweep_configuration = OmegaConf.to_container(sweep_configuration)
    api_key = configs_secret.hyperparameters.wandb_api_key
    wandb.login(key=api_key)

    # Set seed for reproducibility.
    set_seed(seed)
    torch.manual_seed(seed)
    dt = str(datetime.now())[:16]

    data_output_filepath = "data/processed/"

    train_dataset = torch.load(data_output_filepath + "train_dataset.pt")
    eval_dataset = torch.load(data_output_filepath + "eval_dataset.pt")

    sweep_id = wandb.sweep(sweep_configuration, project="mlops_wandb_project")
    if configs_train.hyperparameters.sweeping:
        wandb.agent(sweep_id, function=train, count=10)
    else:
        wandb.agent(sweep_id, function=train, count=1)
    # wandb.watch(model, log_freq=100)

    print("done!")
