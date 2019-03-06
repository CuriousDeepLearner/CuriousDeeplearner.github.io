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

First of all, we will split the dataset into training set and validation set. 
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

def split_train_valid(train:pd.DataFrame, validation_split:float=0.3, shuffle_dataset = True, random_seed = 123):
    """Split the dataset into training set and validation set
    """
    #split train set and valid set
    train_size = len(train)
    idx = list(range(train_size))
    split = int(np.floor(validation_split * train_size)
    #shuffle dataset
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(idx)
    train_idx, val_idx = idx[split:], idx[:split]
    train, val = train.iloc[train_idx], train.iloc[val_idx]
    return train, val

train, val = split_train_valid(train)
#export to csv
train.to_csv('data/train.csv', ignore_index = True)
val.to_csv('data/val.csv', ignore_index=True)
```

Next, we'll create the __tokenizers__. A tokenizer is used to turn a string containing a sentence into a list of individual tokens that make up that string, e.g. "good morning!" becomes ["good", "morning", "!"]. We'll start talking about the sentences being a sequence of tokens from now, instead of saying they're a sequence of words. What's the difference? Well, "good" and "morning" are both words and tokens, but "!" is a token, not a word.

spaCy has model for each language (ex: "en" for English) which need to be loaded so we can access the tokenizer of each model.

> Note: the models must first be downloaded using the following on the command line:
> python -m spacy download en

```python
spacy_en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

Now, TorchText's __Fields__ handle how data should be processed. It holds a Vocab object that defines the set of possible values for elements of the field and their corresponding numerical representations. The Field object also holds other parameters relating to how a datatype should be numericalized, such as a tokenization method and the kind of Tensor that should be produced.

You can read all of the possible arguments [here](https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L61).

```python
TEXT = data.Field(sequential=True, tokenize = tokenizer, lower = True)
LABEL = data.Field(sequential=True, use_vocab= False)
```
The next step is to load training set , validation set, test set from file csv.

```python
train_fields=[('id', None), ('qid', None) , ('question_text', TEXT), ('target', LABEL)]
train, valid = data.TabularDataset.splits('data/', train='train.csv', validation = 'val.csv', 
                                          format='csv', skip_header = True, fields= train_fields)
test_fields = [('qid', None), ('question_text', TEXT)]
test = data.TabularDataset('data/test.csv', format='csv', skip_header=True, fields= test_fields)
```

Next, we'll build the __vocabulary__. The vocabulary is used to associate each unique token with an index (an integer) and this is used to build a one-hot encoding for each token (a vector of all zeros except for the position represented by the index, which is 1). In this case, because we've loaded pretrained embedding, we will convert one-hot encoding to embedding vector (dense vector)

```python
#build vocabulary
TEXT.build_vocab(train, max_size=95000)
```
Once the vocabulary is built, we can obtain different values such as frequency, word index, and the vector representation for each word. The following code demonstrates how to
access these values:
```python
print(TEXT.vocab.freqs)
#sample vocab frequency
Counter({'?': 676528, 
         'the': 326104, 
         'what': 231327, 
         'is': 219286, 
         'a': 200334, 
         'to': 199603, 
         'in': 184911,
         ...
         
print(TEXT.vocab.vectors)
#shape: 95000x300: display the 300 dimension vector for each word
tensor([[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0869,  0.1916,  0.1091,  ..., -0.0152,  0.1111,  0.2065],
        ...,
        [-0.1571,  0.2689,  0.7195,  ...,  0.2269,  0.1317,  0.5180],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]])
         
print(TEXT.vocab.stoi)
# dictionary containing words and their indexes. len = 95003
defaultdict(<function torchtext.vocab._default_unk_index()>,
            {'<unk>': 0,
             '<pad>': 1,
             '?': 2,
             'the': 3,
             'what': 4,
             'is': 5,
             'a': 6,
             'to': 7,
             'in': 8,
             'of': 9,
             'i': 10,
             'how': 11,
```

Now, we are going to load Glove pretrained embedding. The `torchtext` library abstracts away a lot of complexity involved in downloading the embeddings and mapping them to the right word. 

```python
#load pretrained embedding
vectors = torchtext.vocab.Vectors('../data/embeddings/glove.840B.300d/glove.840B.300d.txt')
TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
pretrain_embedding = TEXT.vocab.vectors
```

The final step of preparing the data is to create the iterators.
We also need to define a `torch.device`. This is used to tell TorchText to put the tensors on the GPU or not. We use the `torch.cuda.is_available()` function, which will return `True` if a `GPU` is detected on our computer. We pass this device to the iterator.

When we get a batch of examples using an iterator we need to make sure that all of the question texts are padded to the same length, in training set as well as valid set. Luckily, TorchText iterators handle this for us!

We use a __BucketIterator__ instead of the standard __Iterator__ as it creates batches in such a way that it minimizes the amount of padding in question text. 

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dl, val_dl = data.BucketIterator.splits((train, valid), sort_key = lambda x: len(x.question_text), batch_size = 32 , device = device)
test_dl = data.BucketIterator(test, batch_size = 1, sort=False, shuffle=False, device = device)
``` 

Now, we have `dataloader` for train, valid and test. To create `batch` and display the result of a batch:
```python
batch_train = next(iter(train_dl))
batch_train.question_text

```









 
