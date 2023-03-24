<!-- Table of Contents -->
## Table of Contents

- [Event Relation Classification](#event-relation-classification)
- [CMan](#cman)
- [GPT-3 data augmentation](#gpt-3-data-augmentation)
- [Data](#data)


<!-- Event Relation Classification -->
# Event Relation Classification

This repository contains code and resources for the paper:

> Youssra Rebboud, Pasquale Lisena and Raphaël Troncy.
> **Prompt-based Data Augmentation for
Semantically-precise Event Relation Classification**.
> Submitted at Semantic Methods for Events and Stories (SEMMES) workshop 2023



    pip install -r requirements.txt


## BERT finetuning 
BERT\SpanBERT models fine-tuning for token And sentence classification.
 
<p>To fine-tune BERT models for relation extraction, you can use the <code>sequence_classification.py</code> script. This script takes in a dataset of sentence pairs and their relation labels and fine-tunes a BERT model for relation classification.</p>

<p>To perform inference on the fine-tuned model, you can use the <code>inference.py</code> script. This script takes in a dataset of sentence pairs and performs relation classification using the fine-tuned BERT model.</p>

<p>You can download pre-trained BERT models from the following link:</p>

<ul>
	<li><a href="https://drive.google.com/drive/folders/1R23FpzZr96ugY4qxSs7j7o54D5kL0BI3?usp=sharing">BERT for relation classification</a></li>
</ul>

<p>To fine-tune BERT models for event extraction, you can use the <code>weighted_loss.py</code> script. This script takes in a dataset of sentences and their token labels and fine-tunes a BERT model for event extraction(token classification).</p>


# CMan

Replication of the paper:
> Shan Zhao, Minghao Hu, Zhiping Cai, Fang Liu
> **Modeling Dense Cross-Modal Interactions for Joint Entity-Relation Extraction.**
> In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI)*
> Pages 4032-4038. https://doi.org/10.24963/ijcai.2020/558

Joint relation extraction and entity recognition with a Conditional random field **(CRF)**.

# GPT-3 data augmentation
Scripts for generating new sentences involving event relations, starting from FARO definitions and examples from an existing Dataset.
<h2>Sentence Generation</h2>
<p>To generate new sentences involving event relations, you can use the following command:</p>

<pre><code>python sentence_generation.py</code></pre>

<p>This script will take as input a CSV file containing a list of examples with the desired event relation, and output a new CSV file containing the original data as well as the newly generated sentences.</p>

<h2>Event Triggers Generation</h2>
<p>To generate event triggers for a given set of sentences, you can use the following command:</p>

<pre><code>python Event_triggers_generation_by_GPT-3.py</code></pre>

<p>This script will generate the desired event triggers for the given sentences.</p>

<h2>Cleaning Generated Events</h2>
<p>To clean up the generated event triggers, you can use the following command:</p>

<pre><code>python answers_cleaning.py</code></pre>

<p>This script will clean the generated events from unwanted text using regular expressions output a new CSV file containing the cleaned data.</p>






# Data

The output dataset in csv format, divided in training, test and validation.
