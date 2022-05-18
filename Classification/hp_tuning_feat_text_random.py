"""
This is the main file for hyperparameter tuning of the BERT based classification model 
"""
import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from Data import Data
from Classifier import Classifier
from transformers import DistilBertTokenizer
from transformers import TFDistilBertModel


if __name__=="__main__":
    NUM_ITER = 100 # Number of trainings with different hyperparameter
    
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
    PRE_TRAINED_MODEL_NAME = constants['embeddings']['pretrained_model_name'] #BERT model we choose for contextual embedding extraction
    MODEL = constants['main_params']['model'] # classification or regression head
    MODE = constants['main_params']['mode'] # defines whether the input is text only, text and features or features only
   
    # Defining directories and metrics for saving model weights
    checkpoint_filepath = 'Classification/tmp/checkpoint/feat_text_hp/' +  str(MODEL) + "/"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +"/"
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

    # Initializing tokenizer and bert model 
    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    bert_model = TFDistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # Initializing the Data Class
    data = Data(constants, dataset_variables)    

    # Defining hyperparameters with the value domains, which we randomly sample from
    HP_NUM_LAYER = hp.HParam('num_layer', hp.Discrete([0, 1, 3, 4, 5, 6])) 
    HP_NUM_PRE_LAYER = hp.HParam('num_pre_layer', hp.Discrete([0, 1, 2])) 
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([2, 8, 32, 128, 512])) 
    HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 5e-4, 1e-4, 1e-5])) 
    HP_REG = hp.HParam('regularization', hp.Discrete([0.0, 1e-6, 0.001, 0.005, 0.01, 0.05])) 
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1, 2, 8, 32])) 
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([False, True]))
    HP_DROPOUT_PROP = hp.HParam('dropout_prop', hp.Discrete([0.001, 0.01, 0.05, 0.1, 0.4]))
    HP_BATCH_NORM = hp.HParam('batch_norm', hp.Discrete([False, True]))

    # Defining metrics for evaluating the different models with the hyperparamters
    if MODEL=='regression':
        METRIC = 'mean_absolute_error'
    if MODEL=='classification':
        METRIC = 'accuracy'

    # Define where we save the metrics - hp pairs
    with tf.summary.create_file_writer('Classification/logs/hparam_tuning/feat_text_random/inner_joint').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_LAYER, HP_NUM_PRE_LAYER, HP_NUM_UNITS, HP_LR, HP_REG, HP_BATCH_SIZE, HP_DROPOUT, HP_DROPOUT_PROP, HP_BATCH_NORM],
            metrics=[hp.Metric(METRIC, display_name=METRIC)],
        ) 

    # Start hp tuning for #sessions = NUM_ITER
    for i in range(NUM_ITER):
        #Randomly choose hyperparameters from the domain values provided
        hparams = {
            HP_NUM_LAYER: np.random.choice(HP_NUM_LAYER.domain.values),
            HP_NUM_PRE_LAYER: np.random.choice(HP_NUM_PRE_LAYER.domain.values),
            HP_NUM_UNITS: np.random.choice(HP_NUM_UNITS.domain.values),
            HP_LR: np.random.choice(HP_LR.domain.values),
            HP_REG: np.random.choice(HP_REG.domain.values),
            HP_BATCH_SIZE: np.random.choice(HP_BATCH_SIZE.domain.values),
            HP_DROPOUT: np.random.choice(HP_DROPOUT.domain.values),
            HP_DROPOUT_PROP: np.random.choice(HP_DROPOUT_PROP.domain.values),    
            HP_BATCH_NORM: np.random.choice(HP_BATCH_NORM.domain.values)
            }

        # Define tensorboard callback for visualization                        
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-run-{}_layer-{}_pre-layer-{}_units-{}_lr-{}_reg-{}_bsize-{}_dropout-{}_dropout_prop-{}_bn_{}".format((i+1), 
            hparams[HP_NUM_LAYER], hparams[HP_NUM_PRE_LAYER], hparams[HP_NUM_UNITS], hparams[HP_LR], hparams[HP_REG], 
            hparams[HP_BATCH_SIZE], hparams[HP_DROPOUT], hparams[HP_DROPOUT_PROP], hparams[HP_BATCH_NORM])
        logdir = 'Classification/logs/hparam_tuning/feat_text_random/' + str(MODEL) + "/" + run_name
        tensorboard_callbacks= [
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
            ]

        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
                                    
        # Overwrite classification variables with the sampled ones
        classifier.num_layers_feat_text = hparams[HP_NUM_LAYER]
        classifier.num_pre_layers_feat_text = hparams[HP_NUM_PRE_LAYER]
        classifier.num_hidden_units_feat_text = hparams[HP_NUM_UNITS]
        classifier.learning_rate_feat_text = hparams[HP_LR]
        classifier.regularization_feat_text = hparams[HP_REG]
        classifier.batch_size_feat_text = hparams[HP_BATCH_SIZE]
        classifier.dropout_feat_text = hparams[HP_DROPOUT]
        classifier.dropout_prop_feat_text = hparams[HP_DROPOUT_PROP]
        classifier.batch_norm_feat_text = hparams[HP_BATCH_NORM]

        # Loading the dataset in a DataFrame
        df = pd.read_excel (r'./Data/EMS1.xlsx', sheet_name='EMS 1.0 All 7197 Items KH2 ')
        
        # Cleaning and spliting dataset in its label and input subcomponents 
        df, df_y, df_X_text_list, df_X_feat_list = data.prepare_X_y(df)

        # Splitting the labels in train, val and test
        y_train, y_test, y_val = data.split_data(df_y)

        if MODE=="text&feat" or MODE=="text":
            # Extracting contextualized word embeddings with BERT and splitting them in train, val and test
            X_emb_train_list, X_emb_val_list, X_emb_test_list = data.create_X_emb_list(df, df_X_text_list, df_y, 
                                                                EMBEDDINGS_TYPE, tokenizer, bert_model)

            # Splitting the text string arrays in train, val and test
            df_X_text = pd.concat(df_X_text_list, axis=1)
            X_text_train, X_text_test, X_text_val = data.split_data(df_X_text)
      
        if MODE=="text&feat" or MODE=="feat":
            # Splitting the numerical feature input in train, val and test
            X_feat_train_list, X_feat_val_list, X_feat_test_list, _ = data.create_X_feat_list(df_X_feat_list)

        if data.rebalance==True:
            # Rebalance the train dataset components by deleting samples of the dominant class
            X_text_train, X_emb_train_list, X_feat_train_list, y_train = data.rebalance_f(X_text_train, X_emb_train_list, X_feat_train_list, y_train)

        #Create the datasets for the model from the label array and inout lists of arrays
        train_dataset = data.create_tf_dataset(X_emb_train_list, X_feat_train_list, y_train)
        val_dataset = data.create_tf_dataset(X_emb_val_list, X_feat_val_list, y_val)
        test_dataset = data.create_tf_dataset(X_emb_test_list, X_feat_test_list, y_test)
        
        #Initialize Model for regression. The model is initialized differently depending on its input 
        if MODEL=='regression':
            if MODE=="feat":
                classifier.build_feat_text_regression_model(
                    y_train.shape[1],
                    X_feat_train_list=X_feat_train_list
                    )
            elif MODE=="text":
                classifier.build_feat_text_regression_model(
                    y_train.shape[1], 
                    X_emb_train_list=X_emb_train_list
                    )
            elif MODE=="text&feat":
                classifier.build_feat_text_regression_model(
                    y_train.shape[1], 
                    X_feat_train_list=X_feat_train_list,
                    X_emb_train_list=X_emb_train_list
                    )
        
        #Initialize Model for classification. The model is initialized differently depending on its input 
        elif MODEL=='classification':
            if MODE=="feat":
                classifier.build_feat_text_x_head_classification_model(
                    y_train.shape[1],
                    len(np.unique(y_train)),
                    X_feat_train_list=X_feat_train_list
                    )
            elif MODE=="text":
                classifier.build_feat_text_x_head_classification_model(
                    y_train.shape[1], 
                    len(np.unique(y_train)),
                    X_emb_train_list=X_emb_train_list
                    )
            elif MODE=="text&feat":
                classifier.build_feat_text_x_head_classification_model(
                    y_train.shape[1], 
                    len(np.unique(y_train)),
                    X_feat_train_list=X_feat_train_list,
                    X_emb_train_list=X_emb_train_list
                    )

        #Run training
        classifier.model.fit(train_dataset.batch(classifier.batch_size_feat_text),
                    validation_data=val_dataset.batch(classifier.batch_size_feat_text),
                    epochs=classifier.epochs_feat_text,
                    callbacks=[tensorboard_callbacks, model_checkpoint_callback])
        
        
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