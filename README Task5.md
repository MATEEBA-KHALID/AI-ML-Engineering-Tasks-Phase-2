# Auto Tagging Support Tickets Using LLM

## Objective

Automatically classify support tickets into predefined categories using Large Language Models (LLMs).

## Dataset

Free-text Support Ticket Dataset

Each record contains:

* Ticket Description
* Category Label

## Techniques Used

### Zero-Shot Learning

The model classifies tickets using only instructions and category definitions.

### Few-Shot Learning

The model is provided with example tickets and labels before making predictions.

### Prompt Engineering

Custom prompts are designed to improve classification quality and consistency.

## Model

* Google Flan-T5 Base

## Categories

* Network
* Authentication
* Billing
* Technical
* Account
* Software
* Hardware

## Features

* Zero-shot classification
* Few-shot classification
* Top-3 tag recommendation
* Accuracy comparison
* Classification report generation

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

## Run

pip install -r requirements.txt

python support_ticket_tagger.py

## Skills Gained

* Prompt Engineering
* LLM-based Text Classification
* Zero-shot Learning
* Few-shot Learning
* Multi-class Prediction
* Tag Ranking and Recommendation
