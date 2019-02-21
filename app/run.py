import json
import plotly
import pandas as pd
import numpy as np
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import Counter


app = Flask(__name__)

'''---------Global Variables--------------'''
Stop_Words = nltk.corpus.stopwords.words("english")
Punctuations = str.maketrans('', '', string.punctuation)

def tokenize(text):
    '''
    Tokenizes the text of input
    input: 
        text: text of input

    output: 
        lemmed: returns cleaned tokens in list after removing punctuation, lemmatization, lowered case, and stripped the spaces            
    '''
    text=text.translate(Punctuations)
    token=word_tokenize(text)
    lemmed = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in token if w not in Stop_Words]
    return lemmed
	
def compute_word_counts(messages, filepath='../data/counts.npz'):
    '''
    Function computes the top 20 words in the dataset with counts of each term
    input: 
        messages: list or numpy array
        filepath: filepath to save counts data

    output: 
        top_words: list
        top_counts: list 
    '''
    # get top words 
    counter = Counter()
    for message in messages:
        tokens = tokenize(message)
        for token in tokens:
            counter[token] += 1
            # top 20 words 
            top = counter.most_common(20)
            top_words = [word[0] for word in top]
            top_counts = [count[1] for count in top]
            # save arrays
            np.savez(filepath, top_words=top_words, top_counts=top_counts)
            #to lists
            top_words=list(top_words)
            top_counts=list(top_counts)
            return top_words, top_counts

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # target distribution
    target_distribution = df.drop(['id','message','original','genre'], axis=1).mean()
    target_names = list(target_distribution.index)
    
    # top 20 words and counts 
    top_words, top_counts = compute_word_counts(df.message.values)    
    
    #category names and 
    category_names = df.iloc[:,4:].columns
    category_boolean_counts = (df.iloc[:,4:] != 0).sum().values
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_counts
                )
            ],

            'layout': {
                'title': 'Top 20 words and counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()