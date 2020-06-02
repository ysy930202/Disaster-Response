import sys, pickle, re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine


def load_data(database_filepath):
    table_name = 'Disasters'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    Y = df.iloc[:,4:]
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    words = [WordNetLemmatizer().lemmatize(word, pos='v').lower().strip() for word in words]
    
    return words

def build_model():
    
    """
    Build model with GridSearchCV
    
    Returns:
    Trained model after performing grid search
    
    """
    
    # model pipleline
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {#'tfidf__use_idf': (True, False),
                'clf__estimator__n_estimators': [50, 100]
                #'clf__estimator__min_samples_split': [2, 4]
                } 
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_text)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))
    
def save_model(model, model_filepath):
    """
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the given filepath
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

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