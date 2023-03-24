
  
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Event Relation Classification">About The Project</a>
     
    </li>
    <li>
      
      <ul>
        <li><a href="#BERT finetuning">Prerequisites</a></li>
        <li><a href="#CMan">Installation</a></li>
      </ul>
    </li>
    <li><a href="#GPT-3 data augmentation">Usage</a></li>
    
  </ol>
</details>

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
 

--Relation_extraction: Run sequence_classification.py 

-- inference on relation classification: inference.py 


Link to the model(https://drive.google.com/drive/folders/1R23FpzZr96ugY4qxSs7j7o54D5kL0BI3?usp=sharing)

--Event_extraction: Run weighted_loss.py


# CMan

Replication of the paper:
> Shan Zhao, Minghao Hu, Zhiping Cai, Fang Liu
> **Modeling Dense Cross-Modal Interactions for Joint Entity-Relation Extraction.**
> In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI)*
> Pages 4032-4038. https://doi.org/10.24963/ijcai.2020/558

Joint relation extraction and entity recognition with a Conditional random field **(CRF)**.

# GPT-3 data augmentation

Scripts for generating new sentences involving event relations, starting from definitions and examples.

 --generate_sentences: Run sentence_generation.py
 
 --generate event triggers for given sentences: Run Event_triggers_generation_by_GPT-3.py
 
 --clean generated events: Run answers_cleaning.py

# Data

The output dataset in csv format, divided in training, test and validation.
