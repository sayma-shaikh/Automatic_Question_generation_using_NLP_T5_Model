
import torch
import random
import numpy as np
import textwrap
from transformers import T5ForConditionalGeneration,T5Tokenizer
summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base',model_max_length=1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

# !pip install nltk
from nltk import sent_tokenize
import nltk
nltk.download('punkt')

# """# Keyword generation"""

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback

def get_nouns_multipartite(content):
    out = []
    try:
        # Initialize the extractor
        extractor = pke.unsupervised.MultipartiteRank()

        # Load the document
        extractor.load_document(input=content, language='en')

        # Set the POS tags to include nouns and proper nouns
        pos = {'PROPN', 'NOUN'}

        # Define the stoplist to filter out punctuation marks, stopwords, and numbers
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')

        # Perform candidate selection
        extractor.candidate_selection(pos=pos)

        # Build the Multipartite graph and rank candidates
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')

        # Get the top keyphrases
        keyphrases = extractor.get_n_best(n=25)

        # Extract numbers from the content and add them to the output
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', content)
        out.extend(numbers)

        # Extract keyphrases and add them to the output
        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


import requests  # Add this import statement
import tarfile
from sentence_transformers import SentenceTransformer
from similarity.normalized_levenshtein import NormalizedLevenshtein
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
import numpy as np
nltk.download('wordnet')



normalized_levenshtein = NormalizedLevenshtein()

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1]
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    score = []
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


def get_synonyms(word):
    synonyms = set()
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def get_antonyms(word):
    antonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.append(antonym.name())
    return list(set(antonyms))

def generate_distractors_for_number(word):
    distractors = []
    try:
        # Check if the word contains a decimal point
        if '.' in word:
            # If the word is a decimal number, generate distractors based on decimal values
            integral_part, decimal_part = word.split('.')
            integral_distractors = [str(int(integral_part) - 1), str(int(integral_part) + 1)]
            decimal_distractors = [word.replace(decimal_part, str(int(decimal_part) - 1)),
                                   word.replace(decimal_part, str(int(decimal_part) + 1))]
            distractors.extend(integral_distractors + decimal_distractors)
        else:
            # If the word is a whole number, generate distractors based on integral values
            integral_distractors = [str(int(word) - 1), str(int(word) + 1)]
            distractors.extend(integral_distractors)
    except ValueError:
        # Handle the case if the word is not a valid number
        print("Invalid numeric word:", word)

    return distractors


def generate_distractors_whole(word):
    # Generate distractors for whole numbers based on some logic
    # For example, you could consider factors, multiples, nearby numbers, etc.
    # Here, we'll generate distractors by adding and subtracting some fixed values
    distractors = []
    try:
        number = int(word)
        distractors = [str(number - 1), str(number + 1)]
    except ValueError:
        pass
    return distractors

def generate_distractors_decimal(word):
    # Generate distractors for decimal numbers based on some logic
    # For example, you could consider nearby numbers, numbers with different decimal places, etc.
    # Here, we'll generate distractors by changing the decimal part of the number
    distractors = []
    try:
        integer_part, decimal_part = word.split('.')
        distractors = [word.replace(decimal_part, str(int(decimal_part) - 1)),
                       word.replace(decimal_part, str(int(decimal_part) + 1))]
    except ValueError:
        pass
    return distractors


def get_combined_keyword(distractors_list):
    combined_keyword = ""
    # Iterate through the distractors list for each word
    for distractors in distractors_list:
        if distractors:
            # Select a distractor from the list for each word
            selected_distractor = distractors[0]  # You can choose a distractor based on your preference
            # Combine the selected distractors to form the combined keyword
            combined_keyword += selected_distractor + " "
    # Remove the trailing space and return the combined keyword
    return combined_keyword.strip()


def generate_distractors_for_word(word):
    distractors = []
    try:
         # If the word is not numeric, try to find distractors using WordNet
            syn = wn.synsets(word, 'n')[0]
            word = word.lower()
            orig_word = word
            if len(word.split()) > 0:
                word = word.replace(" ", "_")
            hypernym = syn.hypernyms()
            if len(hypernym) == 0:
                return distractors
            for item in hypernym[0].hyponyms():
                name = item.lemmas()[0].name()
                if name == orig_word:
                    continue
                name = name.replace("_", " ")
                name = " ".join(w.capitalize() for w in name.split())
                if name is not None and name not in distractors:
                    distractors.append(name)
    except:
        print("Error occurred while generating distractors for word:", word)

    return distractors

def is_numeric(word):
    try:
        float(word)
        return True
    except ValueError:
        return False

def generate_numeric_distractors(word):
    # Split the numeric value into its integral and decimal parts
    integral_part, decimal_part = word.split('.')

    # Generate distractors based on the characteristics or context of the numeric value
    # For example, distractors could include related numerical ranges, contextual terms, etc.
    # You can customize this logic based on your specific requirements

    # Example: Generate distractors for the integral part
    integral_distractors = [str(int(integral_part) - 1), str(int(integral_part) + 1)]

    # Example: Generate distractors for the decimal part
    # Assuming decimal part is two digits, generating distractors by changing the digits
    decimal_distractors = [word.replace(decimal_part, str(int(decimal_part) - 1)),
                           word.replace(decimal_part, str(int(decimal_part) + 1))]

    return integral_distractors + decimal_distractors

import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

def get_distractors_wordnet(word):
    distractors = []
    try:
        # print("Attempting to generate distractors for word:", word)
        # Check if the word is a numeric value
        if is_numeric(word):
            # print("Word is a numeric value")
            # Generate numeric distractors
            distractors = generate_numeric_distractors(word)
        else:
            # Attempt to find synsets for the word
            synsets = wn.synsets(word)
            if synsets:
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonym = lemma.name()
                        if synonym != word:
                            distractors.append(synonym)

            # print(distractors)

            # If no synsets are found for the word, attempt to find hypernyms
            if not distractors:
                print("No synsets found in WordNet for individual words, searching for hypernyms")
                hypernyms = get_hypernyms(word)
                if hypernyms:
                    for hypernym in hypernyms:
                        for lemma in hypernym.lemmas():
                            synonym = lemma.name()
                            if synonym != word:
                                distractors.append(synonym)

            # If still no distractors found, return an empty list
            if not distractors:
                print("No synsets or hypernyms found in WordNet")
                return distractors

    except Exception as e:
        print("Error occurred while generating distractors for word:", word)
        print("Error:", e)

    print(distractors)
    return distractors

def is_numeric(word):
    try:
        # Check if the word can be converted to a float
        float(word)
        return True
    except ValueError:
        return False

def generate_numeric_distractors(word):
    distractors = []
    try:
        # Check if the word is a whole number or a decimal number
        if '.' in word:
            integral_part, fractional_part = map(int, word.split('.'))
            # Generate distractors for the fractional part
            for i in range(1, 10):
                modified_fractional_part = str(fractional_part + i)
                distractor = f"{integral_part}.{modified_fractional_part}"
                distractors.append(distractor)
        else:
            number = int(word)
            # Generate distractors by adding/subtracting 1
            distractors = [str(number - 1), str(number + 1)]
            # Generate additional distractors for whole numbers
            while len(distractors) < 4:
                distractors.extend([str(number - len(distractors)), str(number + len(distractors))])
    except ValueError:
        print("Invalid numeric value:", word)
    return distractors

def get_hypernyms(word):
    try:
        # Attempt to find synsets for the word
        synsets = wn.synsets(word)
        if synsets:
            return synsets[0].hypernyms()
        else:
            return None
    except Exception as e:
        print("Error occurred while getting hypernyms for word:", word)
        print("Error:", e)
        return None

def get_distractors(word, orig_sentence, sentence_model, top_n, lambda_val):

    distractors = get_distractors_wordnet(word)
    if not distractors:
        synonyms = get_synonyms(word)
        for synonym in synonyms:
            antonyms = get_antonyms(synonym)
            distractors.extend(antonyms)
    if not distractors:
        return []

    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = orig_sentence + " " + word.capitalize()
    keyword_embedding = sentence_model.encode([embedding_sentence])
    distractor_embeddings = sentence_model.encode(distractors_new)

    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambda_val)

    if filtered_keywords is None:
        return []

    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
            final.append(wrd.capitalize())
    return final[1:]

sent = "What company did Musk say would not accept bitcoin payments?"
keyword = "Tesla"

# Use SentenceTransformer class directly instead of the sentence_transformer_model variable
print(get_distractors(keyword, sent, SentenceTransformer('msmarco-distilbert-base-v3'), 40, 0.2))

# """# MCQs Generation"""

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

# !pip install httpx

import textwrap
import random  # Importing the random module



def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=5,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    question = dec[0].replace("question:", "").strip()
    return question


def generate_question_with_options(text, answer, question_model, question_tokenizer):
    question = get_question(text, answer, question_model, question_tokenizer)
    distractors = get_distractors_wordnet(answer)  # Assuming you have a function get_distractors_wordnet to get distractors
    options = [answer.capitalize()] + distractors[:3]  # Combining answer and distractors for options
    random.shuffle(options)  # Randomly shuffle options

    # Print question
    print(f"Question: {question}?")
    print("\n")

    # Print options
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    print("\n")

    # Print answer
    print(f"Ans: {answer.capitalize()}")
    print("\n")

# """# fill in the blanks"""

import json
import requests
import string
import re
import nltk
import string
import itertools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import json
import requests
import string
import re
import nltk
import string
import itertools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import socket
import huggingface_hub
from huggingface_hub import Repository
import gradio as gr
from gradio import Interface, Textbox, Radio, HTML

context = gr.Textbox(lines=10, placeholder="Enter paragraph/content here...", label="Context")
num_questions_input = gr.Number( label="Number of Questions")
question_type_radio = gr.Radio([ "MCQ", "Fill in the Blanks" ], label="Question Type")
output_box = gr.Textbox(label="Generated Questions")

def tokenize_sentences(context):
    text = context
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

sentences = tokenize_sentences(context)
# print (sentences)

from pprint import pprint
def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(keywords, sentences)
# # pprint (keyword_sentence_mapping_noun_verbs_adj)

import re
def get_fill_in_the_blanks(sentence_mapping):
    out={"title":"Fill in the blanks for these sentences with matching words at the top"}
    blank_sentences = []
    processed = []
    keys=[]
    for key in sentence_mapping:
        if len(sentence_mapping[key])>0:
            sent = sentence_mapping[key][0]

            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods

            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            # if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
            blank_sentences.append(line)
                # processed.append(sentence_mapping[key][0])
            keys.append(key)
    out["sentences"]=blank_sentences  #[:10]
    out["keys"]=keys                  #[:10]

    return out


fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
# pprint(fill_in_the_blanks)

import random

def generate_fill_in_the_blank_question(sentence, correct_answer, distractors, num_options=3):
    options = [correct_answer] + distractors
    random.shuffle(options)

    # Replace the first occurrence of the correct answer with a blank space
    question = sentence.replace(correct_answer, "__________", 1)

    # Add the correct answer and other options
    question_options = []
    for option in options:
        if option.lower() != correct_answer.lower():
            question_options.append(option)
    return (question, question_options, correct_answer)

def generate_FITB_questions(text, num_questions):
    # Get nouns, verbs, and adjectives from the text
    noun_verbs_adj = get_nouns_multipartite(text.lower())

    # Get sentences containing the extracted keywords
    keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, [sentence.lower() for sentence in nltk.sent_tokenize(text)])

    # Initialize a list to store generated questions
    generated_questions = []
    used_keywords = set()

    for keyword, keyword_sentences in keyword_sentence_mapping_noun_verbs_adj.items():
        if keyword.lower() in used_keywords:
            continue
        used_keywords.add(keyword.lower())
        distractors = get_distractors(keyword, " ".join(keyword_sentences), s2v, sentence_transformer_model, 40, 0.2)
        if not distractors:
            continue  # Skip keywords without distractors
        for sentence in keyword_sentences:
            correct_answer = keyword
            question, options, correct_answer = generate_fill_in_the_blank_question(sentence, correct_answer, distractors, num_options=3)

            # Select three distractors randomly
            options = random.sample(distractors, 3)
            options.append(correct_answer)
            random.shuffle(options)

            # Append the question and options as a tuple
            generated_questions.append((question, options, correct_answer))

            if len(generated_questions) == num_questions:
                return generated_questions
                #pass


generated_questions = generate_FITB_questions(text, num_questions)
for i, question_data in enumerate(generated_questions, start=1):
    question, options, correct_answer = question_data
    print(f"Question {i}: {question}\n")
    for j, option in enumerate(options, start=1):
        print(f"{j}. {option}")
    print("\n")


def generate_fill_in_the_blanks(context, num_questions):
    text = context  # Assuming 'context' contains the text input
    generated_questions = generate_FITB_questions(text, num_questions)
    output = ""
    for i, question_data in enumerate(generated_questions, start=1):
        question, options, correct_answer = question_data
        output += f"Question {i}: {question}\n"
        for j, option in enumerate(options, start=1):
            output += f"{j}. {option}\n"
        output += f"\nAns: {correct_answer}\n\n"
    return output


def get_keywords(originaltext): #,summarytext
  keywords = get_nouns_multipartite(originaltext)
  print ("keywords text: ",keywords)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)
  return keywords
keywords = get_keywords(context) #,summarytext


def generate_MCQs_questions(context, num_questions_input, question_model, question_tokenizer):
    np = get_keywords(context) # Assuming get_keywords requires both context and summary_text
    nk = np[:int(num_questions_input)]
    output = ""
    for answer in nk:
        ques = get_question(context, answer, question_model, question_tokenizer) # Changed summary_text to context
        distractors = get_distractors_wordnet(answer)

        # Combine answer and distractors for options
        options = [answer.capitalize()] + distractors[:3]

        # Randomly shuffle options
        random.shuffle(options)

        # Append the answer line at the end of the question
        ques += f"\nAns: {answer.capitalize()}"

        # Join options to the question
        for idx, option in enumerate(options, start=1):
            ques += f"\n{idx}. {option}"

        output += ques + "\n\n"
    return output



def generate_output(context, num_questions, option):
    output = ""

    if option == "MCQ":
        output = generate_MCQs_questions(context, num_questions, question_model, question_tokenizer) # Corrected arguments
    elif option == "Fill in the Blanks":
        output = generate_fill_in_the_blanks(context, num_questions)
    return output

iface = gr.Interface(
    fn=generate_output,
    inputs=[context, num_questions_input, question_type_radio],
    outputs=output_box
)

iface.launch(debug=False)