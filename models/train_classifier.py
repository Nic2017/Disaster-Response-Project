import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import string

'''---------Global Variables--------------'''
Stop_Words = nltk.corpus.stopwords.words("english")
Punctuations = str.maketrans('', '', string.punctuation)

def load_data(database_filepath):
    '''
    Loads data from sqlite database 
    input: 
        database_filepath: path to database

    output: 
        X: features dataframe
        y: target dataframe
        category_names: names of targets in list
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'],axis=1)
    category_names=Y.columns.tolist()
    return X,Y,category_names
    
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


def build_model():
    '''
    build model using Pipeline with TfidfVectorizer&MultiOutputClassifier and Gridsearch
    '''
    pipeline = Pipeline([
    ('TfIdf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
    'TfIdf__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__n_estimators': [10, 20]
    }

    cv = (GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1))
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the trained model using test dataset
    input: 
        model: trained model 
        X_test: Test features 
        Y_test: Test labels 
        category_names: names of lables in list
            
    '''
    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Saves a trained model to a pickle file
    input: 
        model: trained model 
        model_filepath: filepath to save model in binary form          
    '''
    pickle.dump(model,open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()