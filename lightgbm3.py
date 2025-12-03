import pandas as pd
import numpy as np
import logging
import json
import optuna
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, 
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.data.path.append('/home/u493846/nltk_data')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def main():
    
    train = pd.DataFrame.from_records(json.load(open('/home/u493846/train.json')))
    test = pd.DataFrame.from_records(json.load(open('/home/u493846/test.json')))
    train, validation = train_test_split(train, test_size=1/3, random_state=123)
    
    train_fraction = 1.0
    train_reduced = train.sample(frac=train_fraction, random_state=125)
    
    #cleaning the abstract column
    def clean(text):
        #remove urls
        text = re.sub(r'http\S+|www\S+|https\S+', '', text,flags = re.MULTILINE)
        #remove punctuation and special characters
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text
    train_reduced["cleaned_abstract"] = train_reduced["abstract"].apply(clean)
    test["cleaned_abstract"] = test["abstract"].apply(clean)
    validation["cleaned_abstract"] = validation["abstract"].apply(clean)
    
    
    #tokenization
    def tokenize(text):
        return word_tokenize(text)
    train_reduced["tokens"] = train_reduced["cleaned_abstract"].apply(tokenize)
    test["tokens"] = test["cleaned_abstract"].apply(tokenize)
    validation["tokens"] = validation["cleaned_abstract"].apply(tokenize)
    
    #removing stop words
    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower()not in stop_words]
    train_reduced["tokens_no_stop"]= train_reduced["tokens"].apply(remove_stopwords)
    test["tokens_no_stop"]=test["tokens"].apply(remove_stopwords)
    validation["tokens_no_stop"]=validation["tokens"].apply(remove_stopwords)
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'  
        elif tag.startswith('V'):
            return 'v'  
        elif tag.startswith('N'):
            return 'n'  
        elif tag.startswith('R'):
            return 'r'  
        else:
            return 'n'

    def lemmatize(tokens):
        pos_tags = pos_tag(tokens)
        return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    train_reduced["lemmatized_abstract"] = train_reduced["tokens_no_stop"].apply(lambda x: ' '.join(lemmatize(x)))
    test["lemmatized_abstract"] = test["tokens_no_stop"].apply(lambda x: ' '.join(lemmatize(x)))
    validation["lemmatized_abstract"] = validation["tokens_no_stop"].apply(lambda x: ' '.join(lemmatize(x)))
    
    # Topic Modeling
    def add_topic_modeling_features(train_reduced, validation, test):
        vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        train_abstracts = train_reduced['lemmatized_abstract']
        validation_abstracts = validation['lemmatized_abstract']
        test_abstracts = test['lemmatized_abstract']
        
        # Fit on training data
        train_vectors = vectorizer.fit_transform(train_abstracts)
        validation_vectors = vectorizer.transform(validation_abstracts)
        test_vectors = vectorizer.transform(test_abstracts)
        
        # LDA topic modeling
        lda = LatentDirichletAllocation(n_components=10, random_state=123, max_iter=10)
        train_topics = lda.fit_transform(train_vectors)
        validation_topics = lda.transform(validation_vectors)
        test_topics = lda.transform(test_vectors)
        
        # Add topic probabilities as new features
        for i in range(train_topics.shape[1]):
            train_reduced[f'topic_{i}'] = train_topics[:, i]
            validation[f'topic_{i}'] = validation_topics[:, i]
            test[f'topic_{i}'] = test_topics[:, i]
        
        return train_reduced, validation, test

    train_reduced, validation, test = add_topic_modeling_features(train_reduced, validation, test)
    topic_features = [f"topic_{i}" for i in range(10)]
    
    
    #yearly trend citation
    current_year = 2024
    
    train_reduced['years_since_pub'] = current_year - train_reduced['year']
    print(train_reduced[['year', 'years_since_pub']].head())
    validation['years_since_pub'] = current_year - validation['year']
    test['years_since_pub'] = current_year - test['year']

    #handling published papers in current year (avoing zero)
    train_reduced['years_since_pub'] = train_reduced['years_since_pub'].replace(0, 1)
    test['years_since_pub'] = test['years_since_pub'].replace(0, 1)
    validation['years_since_pub'] = validation['years_since_pub'].replace(0, 1)

    # yearly avergae citation
    yearly_avg = train_reduced.groupby('year')['n_citation'].mean().reset_index()
    yearly_avg.rename(columns={'n_citation': 'avg_citations_year'}, inplace=True)


    #merge with original datasets
    train_reduced = train_reduced.merge(yearly_avg, on ='year', how = 'left')
    validation = validation.merge(yearly_avg, on='year', how= 'left')
    test= test.merge(yearly_avg, on ='year', how = 'left')
    
    
    
    #length of lemmatized abstract

    train_reduced['lemmatized_abstract_length'] = train_reduced['lemmatized_abstract'].apply(len)
    validation['lemmatized_abstract_length'] = validation['lemmatized_abstract'].apply(len)
    test['lemmatized_abstract_length'] = test['lemmatized_abstract'].apply(len)
    
    #number of authors per paper
    train_reduced['num_authors'] = train_reduced['authors'].apply(lambda x: len(x.split(',')))
    validation['num_authors'] = validation['authors'].apply(lambda x: len(x.split(',')))
    test['num_authors'] = test['authors'].apply(lambda x: len(x.split(',')))
    
    #readability score of abstract
    def count_syllables(word):
        vowels = "aeiouy"
        word = word.lower()
        count = 0
        if word[0] in vowels:
            count +=1
        for i in range(1, len(word)):
             if word[i] in vowels and word [i-1] not in vowels:
                 count +=1
        if word.endswith('e'):
            count -= 1
        return max(1,count)
    
    def flesch_score(words, abstract):
        sentences = sent_tokenize(abstract)
        syllables = sum(count_syllables(word) for word in words if word.isalpha())
        sentence_count = len(sentences)
        word_count = len(words)
        if sentence_count ==0 or word_count ==0:
            return 0
        asl = word_count / sentence_count #average sentence count
        asw = syllables / word_count
        flesch_score = 206.835 - (1.015 *asl) - (84.6 * asw)
        return flesch_score

    train_reduced['flesch_readability_score'] = train_reduced.apply(lambda row: flesch_score(row['tokens'], row['abstract']),axis = 1)
    test['flesch_readability_score'] = test.apply(lambda row: flesch_score(row['tokens'], row['abstract']),axis = 1)  
    validation['flesch_readability_score'] = validation.apply(lambda row: flesch_score(row['tokens'], row['abstract']),axis = 1)

    
    #feature interaction
    #interaction collaboration and age

    train_reduced['colaboration_trend']= train_reduced['years_since_pub'] * train_reduced['num_authors']
    validation['colaboration_trend']= validation['years_since_pub'] * validation['num_authors']
    test['colaboration_trend']= test['years_since_pub'] * test['num_authors']

    #interaction2 readbility score and abstract length

    train_reduced['readability_length_interaction'] = (
    train_reduced['flesch_readability_score'] * train_reduced['lemmatized_abstract_length']
    )


    validation['readability_length_interaction'] = (
    validation['flesch_readability_score'] * validation['lemmatized_abstract_length']
    )


    test['readability_length_interaction'] = (
    test['flesch_readability_score'] * test['lemmatized_abstract_length']
    )

    #title length
    train_reduced['title_length'] = train_reduced['title'].str.len().fillna(0)
    validation['title_length'] = validation['title'].str.len().fillna(0)
    test['title_length'] = test['title'].str.len().fillna(0)
    
    
 
    
    
    #Target encoding "venue" , replacing each venue with the mean number of citations for papers published in that venue. 
    mean_citations = train.groupby('venue')['n_citation'].mean()
    train_reduced['venue_encoded'] = train_reduced['venue'].map(mean_citations)
    validation['venue_encoded'] = validation['venue'].map(mean_citations)
    test['venue_encoded'] = test['venue'].map(mean_citations)
    
    validation['venue_encoded'] = validation['venue'].map(mean_citations).fillna(mean_citations.mean())
    test['venue_encoded'] = test['venue'].map(mean_citations).fillna(mean_citations.mean())


    #total number of papers cited by each paper
    train_reduced["num_references"] = train_reduced["references"].apply(lambda x: len(x))
    validation["num_references"] = validation["references"].apply(lambda x: len(x))
    test["num_references"] = test["references"].apply(lambda x: len(x))
    
    
    
    featurizer = ColumnTransformer(
        transformers=[ ("years_since_pub", 'passthrough', ["years_since_pub"]),
                       ("venue_encoded", 'passthrough', ["venue_encoded"]),
                       #("venue", 'passthrough', ["venue"]),
                       ("num_authors", 'passthrough', ["num_authors"]),
                       ("num_references", 'passthrough', ["num_references"]),
                       ("lemmatized_abstract_length", 'passthrough', ["lemmatized_abstract_length"]),
                       ("colaboration_trend", 'passthrough', ["colaboration_trend"]),
                       ("readability_length_interaction", 'passthrough', ["readability_length_interaction"]),
                       ("flesch_readability_score", 'passthrough', ["flesch_readability_score"]),
                       ("avg_citations_year", 'passthrough', ["avg_citations_year"]),
                       ("title_length", "passthrough", ["title_length"]),
                       ("authors", TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=2000), "authors"),
                       ("title", TfidfVectorizer(analyzer='word', ngram_range=(1, 3),max_features = 2000), "title"),
                       ("lemmatized_abstract", TfidfVectorizer(analyzer='word', ngram_range=(1, 3),max_features = 3000), "lemmatized_abstract"),
                       
                      ] + [(f"topic_{i}", 'passthrough', [f"topic_{i}"]) for i in range(10)],
        remainder='drop')
    
    densifier = FunctionTransformer(lambda X: X.toarray(), accept_sparse=True)

    dummy= make_pipeline(
    featurizer,
    densifier,
    #SimpleImputer(strategy='mean'),  # Impute missing values with the mean
    DummyRegressor())
#.................................
    #histboost= make_pipeline(
    #featurizer,
    #densifier,
    #HistGradientBoostingRegressor(random_state=42,early_stopping=True))
#.................................

  
    label = "n_citation"
    
    x_validation = validation.drop([label], axis=1)
    y_validation = np.log1p(validation[label])
    
    #  objective function for Optuna
 
    def objective(trial):
        
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.1),
            'random_state': 32,
            'objective': 'regression_l1',
            'metric': 'mae',
            'max_bin': trial.suggest_int('max_bin', 63, 511)
            
        }
        
        lightgbm_model = make_pipeline(
            featurizer,
            #densifier,
            LGBMRegressor(**param)
        )

        #train on rainingset
        X_train = train_reduced.drop([label], axis=1)
        y_train = np.log1p(train_reduced[label])
        lightgbm_model.fit(X_train, y_train)
    
        #Evaluate on validationset
        preds = lightgbm_model.predict(x_validation)
        mae = mean_absolute_error(np.expm1(y_validation), np.expm1(preds))
        
        # Return the validation MAE (to minimize in Optuna)
        return mae

    
    
    study = optuna.create_study(direction='minimize', sampler= optuna.samplers.TPESampler(seed=32))
    study.optimize(objective, n_trials=30, n_jobs=3)
    
    # Output the best parameters and score
    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best score: {-study.best_value}")
    
    # Train the final model using the full training dataset with the best parameters
    X_train = train_reduced.drop([label], axis=1)
    y_train = np.log1p(train_reduced[label])
    
    best_params = study.best_params
    
    fixed_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'random_state': 32,
        #'gpu_use_dp': False
    }
    
    best_params.update(fixed_params)
    best_model = make_pipeline(
        featurizer,
        LGBMRegressor(**best_params)
    )
    best_model.fit(X_train, y_train)

    
   
    for model_name, model in [("dummy", dummy),
                              ("lightgbm", best_model),
                              ]:
        
        logging.info(f"Fitting model {model_name}")
        model.fit(train_reduced.drop([label], axis=1), np.log1p(train_reduced[label].values))
        for split_name, split in [("train     ", train_reduced),
                                  ("validation", validation)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} MAE: {mae:.2f}")
    predicted = np.expm1(best_model.predict(test))
    test['n_citation'] = predicted
    json.dump(test[['n_citation']].to_dict(orient='records'), open('predicted.json', 'w'), indent=2)
        
logging.getLogger().setLevel(logging.INFO)
main()