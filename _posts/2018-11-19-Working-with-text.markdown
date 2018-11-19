---
layout: post
title: "Deep Learning - Working with text"
description: "Introduction of text data. Text data can be seen as either a sequence of characters or a sequence of words"
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


