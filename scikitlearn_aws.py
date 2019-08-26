#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:03:33 2019

@author: jorgeecr
"""

from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil
import string


import argparse
import csv
import json
import numpy as np
import pandas as pd

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

text_column_name = 'intencion'
label_columns_names = ['crec', 'cred', 'equ', 'inic', 'inv', 'mkt', 'no', 'renta', 'sueldo', 'temp']

text_column_dtype = {
        'intencion': str}

label_columns_dtype = {
        'crec': np.int64,
        'cred': np.int64,
        'equ': np.int64,
        'inic': np.int64,
        'inv': np.int64,
        'mkt': np.int64,
        'no': np.int64,
        'renta': np.int64,
        'sueldo': np.int64,
        'temp': np.int64
        }


stopwords = ["acuerdo", "adelante", "ademas","adrede","afirmó","agregó","ahi","ahora"]

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def remove_stopwords(words, stopwords):
        result = [i for i in words if i not in stopwords]
        return result
    
def remove_punctuation(sentence):
        no_punct = ''
        for char in sentence:
            if char not in string.punctuation:
                no_punct = no_punct + char
        return no_punct
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))

    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names= [text_column_name]+label_columns_names,
        dtype=merge_two_dicts(text_column_dtype,label_columns_dtype)) for file in input_files ]
    concat_data = pd.concat(raw_data)
    
    train_texts = concat_data['intencion'].values
    t = list(train_texts)
    t_new = [remove_punctuation(w.lower()) for w in t]
    t_words_split = [w.split() for w in t_new]
    t_nonstop = [remove_stopwords(l,stopwords) for l in t_words_split]
    t_now = [" ".join(l) for l in t_nonstop]
        
    NGRAM_RANGE = (1, 2)
    TOKEN_MODE = 'word'
    MIN_DOCUMENT_FREQUENCY = 2
    
    kwargs = {
                'ngram_range': NGRAM_RANGE,  # Usar 1-gramas y 2-gramas
                'dtype': 'int32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': TOKEN_MODE, 
                'min_df': MIN_DOCUMENT_FREQUENCY,
        }
    vectorizer = TfidfVectorizer(**kwargs)
    
    vectorizer.fit(t_now)
    
    joblib.dump(vectorizer, os.path.join(args.model_dir, "model.joblib"))
  

    def input_fn(input_data, content_type):
        if content_type == 'text/csv':
            df = pd.read_csv(StringIO(input_data),header=None)
            if len(df.columns) == 1:
                df.columns = [text_column_name]
            elif len(df.columns) == 11:
                df.columns = [text_column_name] + label_columns_names
            df = df['intencion']
            return df
        else:
            raise ValueError("{} not supported by script!".format(content_type))

    #input_Data ejemplos: ["este es mi ejemplo"], ["este es otro ejemplo", "de varios casos"]
    
    def predict_fn(input_data, model):
        """Preprocess input data
    
        We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
        so we want to use .transform().
    
        The output is returned in the following order:
    
            rest of features either one hot encoded or standardized
        """
        features = model.transform(input_data)
        features = features.toarray()    
        return features   
       

    def model_fn(model_dir):
        """Deserialize fitted model
        """
        vectorizer = joblib.load(os.path.join(model_dir, "model.joblib"))
        return vectorizer

    def output_fn(prediction, accept):
        """Format prediction output

        The default accept/content-type between containers for serial inference is JSON.
        We also want to set the ContentType or mimetype as the same value as accept so the next
        container can read the response payload correctly.
        """
        if accept == "application/json":
            instances = []
            for row in prediction.tolist():
                instances.append({"features": row})

            json_output = {"instances": instances}

            return worker.Response(json.dumps(json_output), mimetype=accept)
        elif accept == 'text/csv':
            return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
        else:
            raise RuntimeException("{} accept type is not supported by this script.".format(accept))

