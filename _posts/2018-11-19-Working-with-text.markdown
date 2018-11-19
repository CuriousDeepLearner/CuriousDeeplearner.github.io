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

> Example: Consider the sentence "The cat sat on the mat" 
> Set of 2-grams: {"The", "The cat", "cat", "cat sat", "sat", "sat on", "on", "on the", "the", "the mat", "mat"}
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
```







