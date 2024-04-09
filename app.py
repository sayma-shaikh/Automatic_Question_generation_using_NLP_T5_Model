from flask import Flask, request, render_template
import torch
import random
import numpy as np
import textwrap
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Import other necessary libraries

from question_generation import generate_MCQs_questions, generate_fill_in_the_blanks,get_antonyms, get_synonyms ,get_word_definition
from transformers import DistilBertForSequenceClassification


app = Flask(__name__)

# Initialize models and tokenizers
# summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)


# Initialize other components and models

# Define routes
@app.route('/')
def index():
    print("hello world")
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Retrieve input from the form
    context = request.form['context']
    num_questions = int(request.form['num_questions'])
    question_type = request.form['question_type']

    # Generate questions based on the selected type
    output = ""
    if question_type == "MCQ":
        output = generate_MCQs_questions(context, num_questions, question_model, question_tokenizer)
    elif question_type == "Fill in the Blanks":
        output = generate_fill_in_the_blanks(context, num_questions)
    elif question_type == "Definition":
        output = get_word_definition(context)
    elif question_type == "Synonyms":
        output = get_synonyms(context)
    elif question_type == "Antonyms":
        output = get_antonyms(context)

    return render_template('index.html', output=output, context=context, num_questions=num_questions, question_type=question_type)

# Add other necessary functions

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000)    



