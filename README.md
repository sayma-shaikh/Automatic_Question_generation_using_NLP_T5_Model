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

Future improvements include refining the distractor generation process,
enhancing algorithm efficiency, and developing a user-friendly interface for
seamless interaction.

The ultimate goal is to streamline the creation of assessment materials for
educators and content creators.
