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

def preprocess(text):
  def replace_rep(text):
    def _replace_rep(m):
      c,cc = m.groups()
      return f'{c}xxrep'
    re_rep = re.compile(r'(\S)(\1{2,})')
    return re_rep.sub(_replace_rep, text)

  def replace_url(text):
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    return re.sub(URL_PATTERN, 'xxurl', text)

  def replace_emoji(text):
    return emoji_to_thai(text, delimiters=(' ', ' '))

  def replace_punctuation(text):
    punctuation = string.punctuation.replace(".", "") # remove all punctuation except . because we want to keep infomation some word such as "อ.", "จ." and etc.
    punctuation_translator = str.maketrans('', '', punctuation)
    return text.translate(punctuation_translator)

  preprocess_text = text.lower().strip()
  preprocess_text = replace_url(preprocess_text)
  preprocess_text = replace_rep(preprocess_text)
  preprocess_text = replace_punctuation(preprocess_text)
  return preprocess_text

label2id = {'neu': 0, 'pos': 1,'neg': 2,'q': 3}
id2label = {0 :'neu', 1 :'pos', 2 :'neg', 3 :'q'}

current_path = os.getcwd()
model_path = current_path+'\\checkpoint-1000'
config = AutoConfig.from_pretrained(model_path+"\\config.json")
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
model.config.id2label=id2label

tokenizer = AutoTokenizer.from_pretrained(
                                 'airesearch/wangchanberta-base-att-spm-uncased',
                                 revision='main')
                                 

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
    clean_txt = preprocess(text)
    pipe1 = pipeline("text-classification", model=model, tokenizer=tokenizer, device = 0)
    predict_label = pipe1(clean_txt, truncation=True, max_length=512) # X_val is a list from train-test-split
    id = label2id[predict_label[0]['label']]
    pipe2 = pipeline("text-classification", model = model, tokenizer = tokenizer, return_all_scores = True, device = 0)
    predict_prob = pipe2(clean_txt, truncation=True, max_length=512) # X_val is a list from train-test-split
    prob = [[predict_prob[0][0]['score'],predict_prob[0][1]['score'],predict_prob[0][2]['score'],predict_prob[0][3]['score']]]
    result = {"prediction": id2label[id], "Probability": prob,}
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)