import pickle
from flask import Flask, request, jsonify, render_template
import gensim
from gensim.models import KeyedVectors , Word2Vec
import numpy as np
import pandas as pd
import spacy
import string

application = Flask(__name__)
app = application

## Import word embedding model

text_model = KeyedVectors.load('model/model.kv')
ML_model = pickle.load(open('model/randomForest.pkl' , 'rb'))

def sent_vec(sent,model):
    vector_size = model.vector_size
    model_res = np.zeros(vector_size)
    ctr = 1
    for i in sent:
        if i in model.wv:
            ctr += 1
            model_res += model.wv[i]
    model_res = model_res/ctr
    return model_res

#python -m spacy download en_core_web_sm 
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation
def spacy_tokenizer(sent):
    doc = nlp(sent)
    mytokens = [ word.lemma_.lower().strip() for word in doc]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens


## Home page Route
@app.route('/')
def index():
    return render_template('index.html')

## Forms page
@app.route('/predictdata' , methods = ['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        Text = str(request.form.get('Text'))

        string1 = spacy_tokenizer(Text)
        string2 = sent_vec(string1,text_model)
        string2 = string2.reshape(1,-1)
        result1 = ML_model.predict(string2)
        number = result1[0]

        def map_number_to_word(number):
            # Define the mapping dictionary
            mapping_dict = {
                1: 'religion',
                2: 'age',
                3: 'gender',
                4: 'ethnicity',
                5: 'other_cyberbullying',
                0: 'not_cyberbullying'
            }

            # Check if the input number is in the dictionary
            if number in mapping_dict:
                return mapping_dict[number]
            else:
                return 'Unknown'
        
        result = map_number_to_word(number)
        

        return render_template('home.html', result = result)
    
    else:
        return render_template('home.html')
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0")

    




