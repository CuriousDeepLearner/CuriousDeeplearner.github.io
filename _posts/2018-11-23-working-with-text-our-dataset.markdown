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
```

Next, we'll create the __tokenizers__. A tokenizer is used to turn a string containing a sentence into a list of individual tokens that make up that string, e.g. "good morning!" becomes ["good", "morning", "!"]. We'll start talking about the sentences being a sequence of tokens from now, instead of saying they're a sequence of words. What's the difference? Well, "good" and "morning" are both words and tokens, but "!" is a token, not a word.

spaCy has model for each language ("en" for English) which need to be loaded so we can access the tokenizer of each model.

> Note: the models must first be downloaded using the following on the command line:

> python -m spacy download en

```python
spacy_en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenier(text)]
```

Now, TorchText's __Fields__ handle how data should be processed. It holds a Vocab object that defines the set of possible values for elements of the field and their corresponding numerical representations. The Field object also holds other parameters relating to how a datatype should be numericalized, such as a tokenization method and the kind of Tensor that should be produced.

You can read all of the possible arguments [here](https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L61).

```python
TEXT = data.Field(sequential=True, tokenizer = tokenizer, lower = True)
LABEL = data.Field(sequential=True, use_vocab= False)
```
The next step is to load training set , validation set, test set from file csv.

```python
train_fields=[('id', None),( 'question_text', TEXT), ('target', LABEL)]
train, valid = data.TabularDataset.splits('data/', train='train.csv', validation = 'val.csv', format='csv', skip_header = True, fields = train_fields)
test_fields = [('question_text', TEXT)
test = data.TabularDataset('data/test.csv', format='csv', skip_header=True, fields=test_fields)

#load pretrained embedding
vectors = torchtext.vocab.Vectors('../data/embeddings/glove.840B.300d/glove.840B.300d.txt')
```

Next, we'll build the __vocabulary__. The vocabulary is used to associate each unique token with an index (an integer) and this is used to build a one-hot encoding for each token (a vector of all zeros except for the position represented by the index, which is 1). In this case, because we've loaded pretrained embedding, we will convert one-hot encoding to embedding vector (dense vector)

```python
#build vocabulary
TEXT.build_vocab(train, max_size=95000)
TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
pretrain_embedding = TEXT.vocab.vectors
```

The final step of preparing the data is to create the iterators.
We also need to define a `torch.device`. This is used to tell TorchText to put the tensors on the GPU or not. We use the `torch.cuda.is_available()` function, which will return `True` if a `GPU` is detected on our computer. We pass this device to the iterator.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter = data.BucketIterator.splits((train, valid), sort_key = lambda x: len(x.question_text), batch_size = (32, 256), device = device)
test_iter = data.BucketIterator(test, 1, sort=False, shuffle=False, device = device)
``` 







 
