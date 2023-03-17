# Event Relation Classification

This repository contains code and resources for the paper:

> Youssra Rebboud, Pasquale Lisena and Raphaël Troncy.
> **Prompt-based Data Augmentation for
Semantically-precise Event Relation Classification**.
> Submitted at Semantic Methods for Events and Stories (SEMMES) workshop 2023



    pip install -r requirements.txt


## BERT finetuning 
BERT\SpanBERT models fine-tuning for token And sentence classification.
To fine-tune BERT/SpanBERT for token classification, run python sequence_classification.py --model_name_or_path bert-base-cased --task_name ner --do_train --do_eval --data_dir ./data --per_device_train_batch_size 16 --learning_rate 5e-5 --num_train_epochs 3 --output_dir ./output_dir.
To fine-tune BERT/SpanBERT for sentence classification, run python sequence_classification.py --model_name_or_path bert-base-cased --task_name text_classification --do_train --do_eval --data_dir ./data --per_device_train_batch_size 16 --learning_rate 5e-5 --num_train_epochs 3 --output_dir ./output_dir.

# CMan

Replication of the paper:
> Shan Zhao, Minghao Hu, Zhiping Cai, Fang Liu
> **Modeling Dense Cross-Modal Interactions for Joint Entity-Relation Extraction.**
> In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI)*
> Pages 4032-4038. https://doi.org/10.24963/ijcai.2020/558

Joint relation extraction and entity recognition with a Conditional random field **(CRF)**.

# GPT-3 data augmentation

Scripts for generating new sentences involving event relations, starting from definitions and examples.

TBW list of commands for running it

# Data

The output dataset in csv format, divided in training, test and validation.
