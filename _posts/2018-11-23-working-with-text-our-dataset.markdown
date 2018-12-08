---
layout: post
title: "Deep Learning - Working with text - Part 2: Create our own dataset of text"
summary: "Create our own dataset and training "
excerpt: "Working with text and sequence data - Part 1"
date: 2018-11-19
mathjax: true
comments: true
published: true
tags: NLP 
---

This is the second post of series Deep learning - working with text. 

In the [first post](https://curiousdeeplearner.github.io/2018/11/19/Working-with-text/), we were already familiar with some notions and introduction of text as well as build some first models with text. 

In this post, we will together figure out how to create our own dataset for training with text.

## Dataset
Today, we will take a hands-on in an ongoing competition on Kaggle and figure out how we can apply embedding to this competition. The competition is named [__Quora Insinere Questions Classification__](https://www.kaggle.com/c/quora-insincere-questions-classification). In this case, we will be predicting whether a question asked on Quora is sincere or not. And yes, another problem of classification! A quick glance on this problem could remind us another similar one: sentiment analysis of text. 

### Overview of data
The training data includes the question that was asked, and whether it was identified as insincere (target = 1). You could be able to download the dataset from the [official kaggle page](https://www.kaggle.com/c/quora-insincere-questions-classification/data). 

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchtext
from torchtext import data, datasets
import spacy

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#split train set and valid set
train_size = len(train)
idx = list(range(train_size))
split = int(np.floor(validation_split * train_size)
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(idx)
train_idx, val_idx = idx[split:], idx[:split]
x_train, x_val = train.iloc[train_idx], train.iloc[val_idx]
x_train.to_csv('data/train.csv')
x_val.to_csv('data/val.csv')

spacy_en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenier(text)]

TEXT = data.Field(sequential=True, tokenizer = tokenizer, lower = True)
LABEL = data.Field(sequential=True, use_vocab= False)

train_fields=[('id', None),( 'question_text', TEXT), ('target', LABEL)]
train, valid = data.TabularDataset.splits('data/', train='train.csv', validation = 'val.csv', format='csv', skip_header = True, fields = train_fields)
test_fields = [('question_text', TEXT)
test = data.TabularDataset('data/test.csv', format='csv', skip_header=True, fields=test_fields)

#load pretrained embedding
vectors = torchtext.vocab.Vectors('../data/embeddings/glove.840B.300d/glove.840B.300d.txt')
 
#build vocabulary
TEXT.build_vocab(train, max_size=95000)
TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
pretrain_embedding = TEXT.vocab.vectors

train_iter, val_iter = data.BucketIterator.splits((train, valid), sort_key = lambda x: len(x.question_text), batch_size = (32, 256))
test_iter = data.BucketIterator(test, 1, sort=False, shuffle=False)





``` 







 
