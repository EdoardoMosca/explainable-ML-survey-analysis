"""
This is the main file for training and testing the classification model variations 
using an embedding layer instead of BERT
"""
import sys
sys.path.append('./.')
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from sklearn import preprocessing
from Data import Data
from Classifier import Classifier
from embeddings import get_tokens

if __name__=="__main__":
    # Load constants 
    with open('./constants.json') as f:
        constants = json.load(f)

    # Load variable names to select labels, and input features from dataset 
    with open('./dataset_variables.json') as d:
        dataset_variables = json.load(d)

    # Initialize label, numerical features and text feature names
    label_columns = dataset_variables['label_columns'] 
    feature_columns = dataset_variables['feature_columns']
    text_columns = dataset_variables['text_columns']
    
    EMBEDDINGS_TYPE = constants['embeddings']['BERT_embeddings_type'] # Type of embedding used by the model ("CLS", "mean", "BiLSTM")
    MODEL = constants['main_params']['model'] # classification or regression head
    MODE = constants['main_params']['mode'] # defines whether the input is text only, text and features or features only

    # Defining directories for saving training parameters for tensorboard 
    log_dir = "Classification/logs/fit/feat_text_EMB/" +  str(MODEL) + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) 
    
    # Defining directories and metrics for saving model weights
    checkpoint_filepath = 'Classification/tmp/checkpoint/feat_text_EMB/' +  str(MODEL) + "/"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"/"
    if MODEL=='classification':
        monitor = 'val_pred8_accuracy'
        mode = 'max'
    elif MODEL=='regression':
        monitor = 'val_loss'
        mode = 'min'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
        save_weights_only=True, monitor= monitor, mode= mode, save_best_only=True)


    # Initializing the model Classifier class
    classifier = Classifier(constants)
    # Initializing the Data Class
    data = Data(constants, dataset_variables)     
    
    # Loading the dataset in a DataFrame
    df = pd.read_excel (r'./Data/EMS1.xlsx', sheet_name='EMS 1.0 All 7197 Items KH2 ')
    # Cleaning and spliting dataset in its label and input subcomponents 
    df, df_y, df_X_text_list, df_X_feat_list = data.prepare_X_y(df)
    # Splitting the labels in train, val and test
    y_train, y_test, y_val = data.split_data(df_y)
    
    if MODE=="text&feat" or MODE=="text":
        # Extracting word tokens and splitting the token arrays in train, val and test
        df_X_token_list = []
        for text_df in df_X_text_list:
            #Get tokens
            tokens = get_tokens(text_df, classifier.vocab_size_EMB, data.max_sample_length)
            df_X_token_list.append(pd.DataFrame(tokens, index=df.index))
        
        X_token_train_list = []
        X_token_val_list = []
        X_token_test_list = []
        for df_X_token in df_X_token_list:
            X_token_train, X_token_test, X_token_val = data.split_data(df_X_token)

            X_token_train_list.append(X_token_train.copy())
            X_token_val_list.append(X_token_val.copy())
            X_token_test_list.append(X_token_test.copy())
        
        # Splitting the text string arrays in train, val and test
        df_X_text = pd.concat(df_X_text_list, axis=1)
        X_text_train, X_text_test, X_text_val = data.split_data(df_X_text)
    
    if MODE=="text&feat" or MODE=="feat":
        # Splitting the numerical feature input in train, val and test
        X_feat_train_list, X_feat_val_list, X_feat_test_list, _ = data.create_X_feat_list(df_X_feat_list)
    
    if data.rebalance==True:
        # Rebalance the train dataset components by deleting samples of the dominant class
        X_text_train, X_emb_train_list, X_feat_train_list, y_train = data.rebalance_f(X_text_train, X_token_train_list, X_feat_train_list, y_train)
        
    #Create the datasets for the model from the label array and inout lists of arrays
    train_dataset = data.create_tf_dataset(X_token_train_list, X_feat_train_list, y_train)
    val_dataset = data.create_tf_dataset(X_token_val_list, X_feat_val_list, y_val)
    test_dataset = data.create_tf_dataset(X_token_test_list, X_feat_test_list, y_test)
    
    
    #Initialize Model for regression. The model is initialized differently depending on its input 
    if MODEL=='regression':
        if MODE=="feat":
            classifier.build_EMB_feat_text_regression_model(
                y_train.shape[1],
                X_feat_train_list=X_feat_train_list
                )
        elif MODE=="text":
            classifier.build_EMB_feat_text_regression_model(
                y_train.shape[1], 
                X_token_train_list=X_token_train_list
                )
        elif MODE=="text&feat":
            classifier.build_EMB_feat_text_regression_model(
                y_train.shape[1], 
                X_feat_train_list=X_feat_train_list,
                X_token_train_list=X_token_train_list
                )

    #Initialize Model for classification. The model is initialized differently depending on its input 
    elif MODEL=='classification':
        if MODE=="feat":
            classifier.build_EMB_feat_text_x_head_classification_model(
                y_train.shape[1],
                len(np.unique(y_train)),
                X_feat_train_list=X_feat_train_list
                )
        elif MODE=="text":
            classifier.build_EMB_feat_text_x_head_classification_model(
                y_train.shape[1], 
                len(np.unique(y_train)),
                X_token_train_list=X_token_train_list
                )
        elif MODE=="text&feat":
            classifier.build_EMB_feat_text_x_head_classification_model(
                y_train.shape[1], 
                len(np.unique(y_train)),
                X_feat_train_list=X_feat_train_list,
                X_token_train_list=X_token_train_list
                )

    # Run training
    classifier.model.fit(train_dataset.batch(classifier.batch_size_EMB_feat_text),
                validation_data=val_dataset.batch(classifier.batch_size_EMB_feat_text),
                epochs=classifier.epochs_EMB_feat_text,
                callbacks=[tensorboard_callback, model_checkpoint_callback])

    # Evaluate model
    score = classifier.model.evaluate(test_dataset.batch(classifier.batch_size_feat_text), return_dict=True)
    print(f'Score_test: {score}')

    # Load model weights
    classifier.model.load_weights(checkpoint_filepath)

    # Print and save predictions for trianing, validation and test sets
    if MODE=="text&feat" or MODE=="text":
        classifier.plot_and_write_predictions(data, train_dataset, y_train, val_dataset, y_val, test_dataset, y_test, X_text_train, X_text_val, X_text_test)
    else:
        classifier.plot_and_write_predictions(data, train_dataset, y_train, val_dataset, y_val, test_dataset, y_test)
    