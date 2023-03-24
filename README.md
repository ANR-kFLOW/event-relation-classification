<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div>

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
