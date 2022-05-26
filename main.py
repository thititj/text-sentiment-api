from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import re
import torch
import pickle
import string
import warnings
import numpy as np
from pythainlp.util import emoji_to_thai
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tqdm.auto import tqdm
import os

current_path = os.getcwd()
model_path = current_path+'\\checkpoint-500'
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4)
tokenizer = AutoTokenizer.from_pretrained(
                                 'airesearch/wangchanberta-base-att-spm-uncased',
                                 revision='main')
                                 
test_trainer = Trainer(model)

app = FastAPI()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


@app.get("/")
def read_root():
    return {
        "API": "Wisesight sentiment classification",
        "version": "0.1.0",
        "author": "Pakin Siwatammarat",
        "docs": "/docs"
    }

@app.post('/predict/')
def predict_sentiment(text: str):
    tokenize_sentence = tokenizer([text], padding=True, truncation=True, max_length=512)
    dataset_sentence = Dataset(tokenize_sentence)
    raw_pred = test_trainer.predict(dataset_sentence)
    pred =  np.argmax(raw_pred.predictions)
    prob=torch.softmax(torch.tensor(raw_pred.predictions),dim=1).numpy().tolist()
    sentiments = {0: "Neutral", 1: "Positive", 2: "Negative", 3: "Question"}
    result = {"prediction": sentiments[pred], "Probability": prob}
    return result


