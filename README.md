---
title: Automatic Question Generation Using NLP
emoji: 📄✏️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 4.25.0
app_file: app.py
pinned: false
short_description: generating questions from text
---

# Automatic_Question_generation_using_NLP_T5_Model

The project aims to automate the generation of distractors for multiple-choice
questions.

It employs WordNet for word sense disambiguation and a sentence transformer
model for sentence embedding.

Cosine similarity is utilized to identify relevant distractors based on semantic
similarity with the keyword.

Compatibility issues between `httpx` and `httpcore` currently hinder
functionality.
