---
layout: post
title: "Deep Learning - Working with text"
summary: "Introduction of text data. Text data can be seen as either a sequence of characters or a sequence of words"
excerpt: "Working with text and sequence data - Part 1"
date: 2018-11-19
mathjax: true
comments: true
published: true
tags: NLP 
---

This post is the first in a series about working with text data

## Introduction

Text is one of the commonly used sequential data types. Text data can be seen as either a sequence of characters or a sequence of words. It is common to see text as a sequence of words for most problems. Deep learning sequential models such as RNN and its variants are able to learn important patterns from text data that can solve problems in areas such as: 
- Natural language understanding
- Document classification 
- Sentiment classification

Applying deep learning to text is a fast-growing field, and a lot of new techniques arrive every month. We will cover the fundamental components that power most of the modern-day deep learning applications.

Deep learning models, like any other machine learning model, do not understand text, so we need to convert text into numerical representation. The process of converting text into numerical representation is called __vectorization__ and can be done in different ways,: 

- Convert text into words and represent each word as a vector
- Convert text into characters and represent each character as a vector
- Create n-gram of words or characters and transform them as vectors. N-grams are overlapping groups of multiple consecutive words or characters.

Text data can be broken down into one of these representations. Each smaller unit of text is called __token__, and the process of breaking text into tokens is called __tokenization__. Once we convert the text data into tokens, we then need to map each token to a vector. __One-hot Encoding__ and __Words Embedding__ are the two most popular approaches for mapping tokens to vectors.

### Understanding n-grams and bag-of-words
Word n-grams are groups of n(or fewer) consecutive words that you can extract from a sentence. The same concept may also be applied to characters instead of words. 

> Example: Consider the sentence "The cat sat on the mat" \\
> Set of 2-grams: {"The", "The cat", "cat", "cat sat", "sat", "sat on", "on", "on the", "the", "the mat", "mat"} \\
> Set of 3-grams: {"The", "The cat", "The cat sat", "cat", "cat sat", "cat sat on", "sat", "sat on", "sat on the","on", "on the mat", "the", "the mat", "mat"}

Such a set is called a bag-of-2-grams or bag-of-3-grams, respectively. The term bag here refers to the fact that you're dealing with a set of tokens rather than a list of sequence: the tokens have no specific order. This family of tokenization methods is called __bag-of-words__.

Because __bag-of-words__ isn't an order-preserving tokenization method, it tends to be used as a form of feature engineering in shallow language-processing models(logistic regression, random forests...) rather than in deep learning models (convnets, recurrent neural networks...).

```python
from nltk import ngrams
#list of 2 grams
2-grams = list(ngrams(text.split(), 2))
```

### Vectorization
#### One-hot encoding
In one-hot encoding, each token is represented by a vector of length N, where N is the size of the vocabulary. The vocabulary is the total number of unique words in the document.

> Consider the sentence: An apple a day keeps doctor away said the doctor

```python
import numpy as np
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
	self.idx2word = []
	self.length = 0
   
    def add_word(self, word):
	if word not in self.idx2word:
	     self.idx2word.append(word)
	     self.word2idx[word] = self.length + 1
	     self.length += 1
	return self.word2idx[word]

    def __len__(self):
	return len(self.idx2word)

    def onehot_encoded(self, word):
	vec = np.zeros(self.length)
	vec[self.word2idx[word]] = 1
	return vec

dic = Dictionary()
for tok in text.split():
    dic.add_word(tok)

dic.onehot_encoded('day') 	
```

One of the challenges with one-hot representation is that the data is too sparse, and the size of the vector quickly grows as the number of unique words in the vocabulary increases, which is considered to be a limitation, and hence it is rarely used with deep learning. 

A variant of one-hot encoding is the so-called __one-hot hashing trick__, which you can use when the number of unique tokens in your vocabulary is too large to handle explicitly. Instead of explicitly assigning an index to each word and keeping a reference of these indices in a dictionary, you can hash words into vectors of fixed size. This is typically done with a very lightweight hashing function. The main advantage of this method is that it does away with maintaining an explicit word index, which saves memory and allows online encoding of the data. The one drawback of this appoach is that it's susceptible to hash collisions: two different words may end up with the same hash, and subsequently any machine learning model looking at these hashes won't be able to tell the difference between these words.

#### Word embedding

Whereas the vectors obtained through one-hot encoding are binary, sparse and very high-dimensional, word embeddings are low dimensional floating-point vectors(dense vectors as opposed to sparse vectors). Morever, unlike the word vectors obtained via one-hot encoding, word embeddings are learned from data via training. It's common to use a word embedding of dimension size 50, 100, 256 or 300. This dimension size is a __hyper-parameter__ that we need to play with during the training phase. 

There are two ways to obtain word embeddings: 

- Learn word embeddings jointly with the main task you care about. It means that you start with random word vectors and then learn word vectors in the same way you learn the weights of a neural network

- Load into your model word embeddings that were precomputed using a different machine-learning task than the one you're trying to solve. These are called __pretrained word embeddings__

##### Learning word embedding with the Embedding Layer

One way to create a word embedding is to start with dense vector for each token containing random numbers, and then train a model such as a document classifier or sentiment classifier. The floating point numbers, which represent the tokens, will get adjusted in a way such that semantically closer words will have similar representations. For instance, in a reasonable embedding space, you would expect synonyms to be embedded into similar word vectors

The __Embedding__ layer is best understood as a dictionary that maps integer indices (which stand for specific words) to dense vectors. It takes integers as input, it look up these integers in an internal dictionary, and it returns the associated vectors. It's effectively a dictionary lookup. 

```python
#pytorch 
import torch.nn as nn

word_to_ix = {'hello': 0, 'world': 1}

embeds = nn.Embedding(2, 5) #2 words in vocab, 5: dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix['hello']], dtype= torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
##tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]], grad_fn=<EmbeddingBackward>)

#Keras
from keras.layers import Embedding
embedding_layer = Embedding(2, 5)
```

> When you instantiate an Embedding layer, its weights are initially random, just as with any other layer.

##### Using Pretrained Word Embeddings
when we have little data on which we cannot meaningfully train the embeddings, we can use embeddings, which are trained on different data corpuses such as Wikipedia, Google, News and Twitter tweets. That embeddings vectors is highly structured and exhibits useful properties-that captures generic aspects of language structure. The rationale behind using pretrained word embeddings in NLP is much the same as for using pretrained convnets in image classification in computer vision: We don't have enough data available to learn truly powerful features on our own, but we expect the features that we need to be fairly generic-that is, common visual features or semantic features. 

There are various pretrained word embeddings available for download:
1. GloVe (Global Vectors for Words Representation,[stanford](https://nlp.stanford.edu/projects/glove)): developed by Stanford researchers in 2014. 
2. Word2vec (Google News vectors, [google](https://code.google.com/archive/p/word2vec)): developed by Google in 2013
3. fasttext (wiki news, [fasttext](https://fasttext.cc/docs/en/english-vectors.html))
4. CharNGram









