""" 
This script loads a model and evalutes that model.
config file: predict.yaml
"""
# Misc
#from select import EPOLLEXCLUSIVE
import numpy as np
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
from omegaconf import OmegaConf

# Hyperparameters extracted
config_path="./configs/"

configs_predict = OmegaConf.load(config_path+'predict.yaml')
path  = configs_predict.hyperparameters.model_filepath
model_name = configs_predict.hyperparameters.model

class ElectraClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(ElectraClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel.from_pretrained("google/electra-small-discriminator")
        self.dense1 = nn.Linear(self.electra.config.hidden_size, self.electra.config.hidden_size)
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
        discriminator_hidden_states = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        return logits


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


def get_predictions(model, data_loader):
    """Computes predections and predection probabilities with input model and dateloader"""
    model = model.eval()
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()

    return predictions, prediction_probs, real_values


def load_model(ElectraClassifier,path,model_name):
    model = ElectraClassifier()
    prams=torch.load(path+model_name)
    model.load_state_dict(prams)
    return model

def get_classification_report(predections,true):
    print(classification_report(predections, true,digits=4))



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_output_filepath = "data/processed/"
    model = load_model(ElectraClassifier,path,model_name)
    test_dataset = torch.load(data_output_filepath + "test_dataset.pt")
    test_dataloader = DataLoader(test_dataset, batch_size=2)
    predictions, prediction_probs, real_values = get_predictions(model, test_dataloader)
    get_classification_report(predictions,real_values)
    # print(prediction_probs)

