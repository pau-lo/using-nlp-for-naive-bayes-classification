# using-nlp-for-naive-bayes-classification

Using natural language processing (nlp) and naive bayes classifier for determining sms messages as 'spam' or 'ham'.

## SMS-Spam-Classifier App prototype example

<div style="text-align:center">
    <img src ="https://github.com/pau-lo/using-nlp-for-naive-bayes-classification/blob/main/SMS-Spam-Classifier-App-master/ham-or-spam-nb-classifier/figures/img/app.PNG"></div>

## Overview

SMS Spam Collection Dataset
Collection of SMS messages tagged as spam or legitimate

**Data description**

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

Data source: [kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset?select=spam.csv) by UCI Machine Learning Repository.  Version 1.

**Overview of ML Process:**

- Business understanding: Not enable in this prototype app.
- Data understanding
- Data preparation
- Modeling
- Evaluation
- Deployment

## Question to answer:

Can you use this dataset to build a prediction model that will accurately classify which texts are spam or ham?

- This model will classifiy whether a text is spam or ham.
- Will make a small flask app prototype to let users and stakeholders test it out on the browser.

## Dependencies

run the requirements.txt file as follow:

    pip install -r /path/to/requirements.txt 


## Basic Usage

To run, type the following into the terminal/bash or on windows shell/cmd:

    python app.py 
    


