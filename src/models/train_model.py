# Misc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

import random
from tqdm.auto import tqdm
from collections import defaultdict

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
    AutoTokenizer,
    AutoModelForPreTraining,
    AdamW,
    get_scheduler,
    get_linear_schedule_with_warmup,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pdb

# from torchviz import make_dot


from transformers import ElectraModel, TrainingArguments, Trainer, AutoTokenizer
import torch
from torch import nn


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
        pretrained_model_name_or_path="google/electra-small-discriminator", num_labels=2
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
        self.dropout1 = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra.config.hidden_size, self.num_labels)

    def classifier(self, sequence_output):
        x = sequence_output[:, 0, :]
        # x = self.dropout1(x)
        x = F.gelu(self.dense1(x))
        x = self.dropout2(x)
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
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


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


def train_new(model, train_dataset, eval_dataset):
    history = defaultdict(list)
    best_accuracy = 0
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

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), "best_model_state.bin")
            best_accuracy = val_acc


# def run(data_output_filepath):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = load_model(device)
#     train_dataset = torch.load(data_output_filepath + "train_dataset.pt")
#     eval_dataset = torch.load(data_output_filepath + "eval_dataset.pt")
#     train(model, train_dataset, eval_dataset)


if __name__ == "__main__":
    data_output_filepath = "data/processed/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ElectraClassifier()
    model = model.to(device)
    train_dataset = torch.load(data_output_filepath + "train_dataset.pt")
    eval_dataset = torch.load(data_output_filepath + "eval_dataset.pt")
    train_dataloader = DataLoader(train_dataset)
    # pdb.set_trace()
    eval_dataloader = DataLoader(eval_dataset)

    EPOCHS = 1
    optimizer = AdamW(
        model.parameters(), lr=2e-5, correct_bias=False, no_deprecation_warning=True
    )
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    train_new(model, train_dataset, eval_dataset)
