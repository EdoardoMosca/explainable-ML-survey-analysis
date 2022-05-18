"""
Classifier class enclosing all functions related model creating for the 
different model variations
"""
import sys
sys.path.append('./.')
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from data_analysis import plot_confusion_matrix, plot_Q20_pred_label_distr
from transformers import TFDistilBertModel


class Classifier():
    def __init__(self, constants):

        # Initializing the model variables ...
        self.constants = constants
        self.MODEL = self.constants['main_params']['model'] # classification or regression head
        self.MODE = self.constants['main_params']['mode'] # defines whether the input is text only, text and features or features only
        self.EMBEDDINGS_TYPE = constants['embeddings']['BERT_embeddings_type'] # Type of embedding used by the model ("CLS", "mean", "BiLSTM")
        
        # ...for the BERT based model 
        self.regularization_feat_text = self.constants['hyper_params_feat_text_classifier']['regularization']
        self.batch_norm_feat_text = self.constants['hyper_params_feat_text_classifier']['batch_norm']
        self.dropout_feat_text = self.constants['hyper_params_feat_text_classifier']['dropout']
        self.dropout_prop_feat_text = self.constants['hyper_params_feat_text_classifier']['dropout_prop']
        self.num_layers_feat_text = self.constants['hyper_params_feat_text_classifier']['num_layers']
        self.num_pre_layers_feat_text = self.constants['hyper_params_feat_text_classifier']['num_pre_layers']
        self.num_hidden_units_feat_text = self.constants['hyper_params_feat_text_classifier']['num_hidden_units']
        self.learning_rate_feat_text = self.constants['hyper_params_feat_text_classifier']['learning_rate']
        self.batch_size_feat_text = self.constants['hyper_params_feat_text_classifier']['batch_size']
        self.epochs_feat_text = self.constants['hyper_params_feat_text_classifier']['epochs']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate_feat_text,
            decay_steps=3000,
            decay_rate=0.9)
        self.optimizer_feat_text = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # ...for the EMBedding layer based model 
        self.regularization_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['regularization']
        self.batch_norm_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['batch_norm']
        self.dropout_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['dropout']
        self.dropout_prop_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['dropout_prop']
        self.num_layers_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['num_layers']
        self.num_pre_layers_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['num_pre_layers']
        self.num_hidden_units_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['num_hidden_units']
        self.learning_rate_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['learning_rate']
        self.batch_size_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['batch_size']
        self.epochs_EMB_feat_text = self.constants['hyper_params_EMB_feat_text_classifier']['epochs']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate_EMB_feat_text,
            decay_steps=3000,
            decay_rate=0.9)
        self.optimizer_EMB_feat_text = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.vocab_size_EMB = self.constants['hyper_params_EMB_feat_text_classifier']['vocab_size']
        self.embedding_dim_EMB = self.constants['hyper_params_EMB_feat_text_classifier']['embbeding_dim']

        # ...for both models (general parameters)
        self.initialization = tf.keras.initializers.GlorotUniform()
        self.loss_classif = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics_classif = [self.constants['general_classifier_params']['metrics_classifier']]
        self.loss_reg = self.constants['general_classifier_params']['loss_regression']
        self.metrics_reg = [self.constants['general_classifier_params']['metrics_regression']]


    def build_feat_text_regression_model(self, num_heads, X_feat_train_list=None, X_emb_train_list=None):
        """
        Builds the BERT based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): The number of prediction heads - 8 in our case for the 8 career goals
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics
            X_emb_train_list (list of np.arrays): list with n arrays containing the text embeddings (N x 768) for each of the n text variables

        Returns (Initializes):
            self.model: model we want to return
        """
        
        if self.MODE=="feat" or self.MODE=="text&feat":
            #Initialize input with numerical features
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            #If num_pre_layers_feat_text > 0 we process the numerical features in different streams depending on the topic they belong to
            if int(self.num_pre_layers_feat_text) > 0: 
                x_i_list = []
                # Each topic gets proessed independently
                for i in range(len(X_feat_train_list)):
                    #First layer with dropout and batch_norm optional
                    x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_feat_text))(inputs[i])
                    if self.batch_norm_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)
                    
                    #Following layer with dropout and batch_norm optional
                    for i in range(int(self.num_pre_layers_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_feat_text))(x_i)
                        if self.batch_norm_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)
                    #Creating a list of stream outputs which we concatenate later
                    x_i_list.append(x_i) 
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
        
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
        
        if self.MODE=="text&feat" or self.MODE=="text":
            if self.EMBEDDINGS_TYPE=="BiLSTM":
                for i in range(len(X_emb_train_list)):
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1], X_emb_train_list[i].shape[2]), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    x = Bidirectional(LSTM(512))(inputs_text)
                    out.append(x)
            else:
                for i in range(len(X_emb_train_list)):
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1],), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    out.append(inputs_text)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)

        for i in range(int(self.num_layers_feat_text)):
            x = Dense(self.num_hidden_units_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text))(x)
            if self.batch_norm_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_feat_text==True:
                x = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x)

        outputs = Dense(num_heads, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text), name="predictions")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(optimizer=self.optimizer_feat_text,                                     
                    loss=self.loss_reg,
                    metrics=self.metrics_reg)

    def build_feat_text_x_head_classification_model(self, num_heads, num_classes, X_feat_train_list=None, X_emb_train_list=None):
        """
        Builds the BERT based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): number of prediction heads - 8 in our case for the 8 career goals
            num_classes (int): number of classes (distinct prediction values)/head
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics
            X_emb_train_list (list of np.arrays): list with n arrays containing the text embeddings (N x 768) for each of the n text variables

        Returns (Initializes):
            self.model: model we want to return
        """
        outputs = []
        if self.MODE=="feat" or self.MODE=="text&feat":
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            if int(self.num_pre_layers_feat_text) > 0: 
                x_i_list = []
                for i in range(len(X_feat_train_list)):
                    x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_feat_text))(inputs[i])
                    if self.batch_norm_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)

                    for i in range(int(self.num_pre_layers_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_feat_text))(x_i)
                        if self.batch_norm_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)
                    
                    x_i_list.append(x_i)
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
       
        if self.MODE=="text&feat" or self.MODE=="text":
            if self.EMBEDDINGS_TYPE=="BiLSTM":
                for i in range(len(X_emb_train_list)):
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1], X_emb_train_list[i].shape[2]), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    x = Bidirectional(LSTM(512))(inputs_text)
                    out.append(x)
            else:
                for i in range(len(X_emb_train_list)):
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1],), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    out.append(inputs_text)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)
        
        for i in range(int(self.num_layers_feat_text)):
            x = Dense(self.num_hidden_units_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_feat_text))(x)
            if self.batch_norm_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_feat_text==True:
                x= tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x)

        for i in range(num_heads):
            output = Dense(num_classes, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text), name="pred"+str(i+1))(x)
            outputs.append(output)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer_EMB_feat_text,                                     
                    loss=[self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif],
                    metrics=[self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif],)
    
    def build_EMB_feat_text_regression_model(self, num_heads, X_feat_train_list=None, X_token_train_list=None):
        """
        Builds the embedding layer based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): The number of prediction heads - 8 in our case for the 8 career goals
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics
            X_emb_train_list (list of np.arrays): list with n arrays containing the text embeddings (N x 768) for each of the n text variables

        Returns (Initializes):
            self.model: model we want to return
        """
        if self.MODE=="feat" or self.MODE=="text&feat":
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            if int(self.num_pre_layers_EMB_feat_text) > 0: 
                x_i_list = []
                for i in range(len(X_feat_train_list)):
                    x_i = Dense(int(self.num_hidden_units_EMB_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(inputs[i])
                    if self.batch_norm_EMB_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_EMB_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x_i)

                    for i in range(int(self.num_pre_layers_EMB_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_EMB_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(x_i)
                        if self.batch_norm_EMB_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_EMB_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x_i)
                    
                    x_i_list.append(x_i) 
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
        
        if self.MODE=="text&feat" or self.MODE=="text":
            for i in range(len(X_token_train_list)):
                inputs_text = tf.keras.layers.Input(shape=(X_token_train_list[i].shape[1],), name="embeddings"+str(i+1))
                x = tf.keras.layers.Embedding(self.vocab_size_EMB, self.embedding_dim_EMB, input_length=X_token_train_list[i].shape[1])(inputs_text)
                x = tf.keras.layers.GlobalMaxPooling1D()(x)
                
                inputs.append(inputs_text)
                out.append(x)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)

        for i in range(int(self.num_layers_EMB_feat_text)):
            x = Dense(self.num_hidden_units_EMB_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_EMB_feat_text))(x)
            if self.batch_norm_EMB_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_EMB_feat_text==True:
                x = tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x)

        outputs = Dense(num_heads, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_EMB_feat_text), name="predictions")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer_EMB_feat_text,                                     
                    loss=self.loss_reg,
                    metrics=self.metrics_reg)


    def build_EMB_feat_text_x_head_classification_model(self, num_heads, num_classes, X_feat_train_list=None, X_token_train_list=None):
        """
        Builds the embedding layer based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): The number of prediction heads - 8 in our case for the 8 career goals
            num_classes (int): number of classes (distinct prediction values)/head
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics
            X_emb_train_list (list of np.arrays): list with n arrays containing the text embeddings (N x 768) for each of the n text variables

        Returns (Initializes):
            self.model: model we want to return
        """
        outputs = []
        if self.MODE=="feat" or self.MODE=="text&feat":
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            if int(self.num_pre_layers_EMB_feat_text) > 0: 
                x_i_list = []
                for i in range(len(X_feat_train_list)):
                    x_i = Dense(int(self.num_hidden_units_EMB_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(inputs[i])
                    if self.batch_norm_EMB_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_EMB_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x_i)

                    for i in range(int(self.num_pre_layers_EMB_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_EMB_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(x_i)
                        if self.batch_norm_EMB_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_EMB_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x_i)
                    
                    x_i_list.append(x_i)
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy() 
        
        if self.MODE=="text&feat" or self.MODE=="text":
            for i in range(len(X_token_train_list)):
                inputs_text = tf.keras.layers.Input(shape=(X_token_train_list[i].shape[1],), name="embeddings"+str(i+1))
                x = tf.keras.layers.Embedding(self.vocab_size_EMB, self.embedding_dim_EMB, input_length=X_token_train_list[i].shape[1])(inputs_text)
                x = tf.keras.layers.GlobalMaxPooling1D()(x)
                
                inputs.append(inputs_text)
                out.append(x)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)

        for i in range(int(self.num_layers_EMB_feat_text)):
            x = Dense(self.num_hidden_units_EMB_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_EMB_feat_text))(x)
            if self.batch_norm_EMB_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_EMB_feat_text==True:
                x = tf.keras.layers.Dropout(self.dropout_prop_EMB_feat_text)(x)

        for i in range(num_heads):
            output = Dense(num_classes, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_EMB_feat_text), name="pred"+str(i+1))(x)
            outputs.append(output)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer_EMB_feat_text,                                     
                    loss=[self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif],
                    metrics=[self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif],)

    def build_one_flow_feat_text_regression_model(self, num_heads, pre_trained_model_name, data, X_feat_train_list=None):
        """
        Builds the one flow BERT based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): The number of prediction heads - 8 in our case for the 8 career goals
            pre_trained_model_name (string): name of the BERT model we use
            data (object): object from Data class
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics

        Returns (Initializes):
            self.model: model we want to return
        """
        if self.MODE=="feat" or self.MODE=="text&feat":
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            if int(self.num_pre_layers_feat_text) > 0: 
                x_i_list = []
                for i in range(len(X_feat_train_list)):
                    x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(inputs[i])
                    if self.batch_norm_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)

                    for i in range(int(self.num_pre_layers_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(x_i)
                        if self.batch_norm_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)
                    
                    x_i_list.append(x_i)
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy() 
        
        if self.MODE=="text&feat" or self.MODE=="text":
            
            inputs_text_ids = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_ids", dtype='int32')
            inputs_text_am = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_am", dtype='int32')                       
            bert_model = TFDistilBertModel.from_pretrained(pre_trained_model_name)
            bert_model.trainable = False
            X_embeddings = bert_model([inputs_text_ids, inputs_text_am])[0]

            if self.EMBEDDINGS_TYPE=="mean":
                X_embeddings = tf.math.reduce_mean(X_embeddings, axis=1)
            elif self.EMBEDDINGS_TYPE=="CLS":
                X_embeddings = tf.squeeze(X_embeddings[:, 0:1, :], axis=1)
            
            inputs.append(inputs_text_ids)
            inputs.append(inputs_text_am)
            out.append(X_embeddings)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)

        for i in range(int(self.num_layers_feat_text)):
            x = Dense(self.num_hidden_units_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text))(x)
            if self.batch_norm_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_feat_text==True:
                x = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x)

        outputs = Dense(num_heads, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text), name="predictions")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer_feat_text,                                     
                    loss=self.loss_reg,
                    metrics=self.metrics_reg)
    
    def build_one_flow_feat_text_classification_model(self, num_heads, num_classes, pre_trained_model_name, data, X_feat_train_list=None):
        """
        Builds the one flow BERT based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            num_heads(int): The number of prediction heads - 8 in our case for the 8 career goals
            num_classes (int): number of classes (distinct prediction values)/head
            pre_trained_model_name (string): name of the BERT model we use
            data (object): object from Data class
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics

        Returns (Initializes):
            self.model: model we want to return
        """
        outputs = []
        if self.MODE=="feat" or self.MODE=="text&feat":
            inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
            
            if int(self.num_pre_layers_feat_text) > 0: 
                x_i_list = []
                for i in range(len(X_feat_train_list)):
                    x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(inputs[i])
                    if self.batch_norm_feat_text==True:
                        x_i = tf.keras.layers.BatchNormalization()(x_i)
                    if self.dropout_feat_text==True:
                        x_i= tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)

                    for i in range(int(self.num_pre_layers_feat_text-1)):
                        x_i = Dense(int(self.num_hidden_units_feat_text/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(self.regularization_EMB_feat_text))(x_i)
                        if self.batch_norm_feat_text==True:
                            x_i = tf.keras.layers.BatchNormalization()(x_i)
                        if self.dropout_feat_text==True:
                            x_i = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x_i)
                    
                    x_i_list.append(x_i) 
                    out = x_i_list.copy()
            else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()

        if self.MODE=="text&feat" or self.MODE=="text":
            inputs_text_ids = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_ids", dtype='int32')
            inputs_text_am = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_am", dtype='int32')                       
            bert_model = TFDistilBertModel.from_pretrained(pre_trained_model_name)
            bert_model.trainable = False
            X_embeddings = bert_model([inputs_text_ids, inputs_text_am])[0]

            if self.EMBEDDINGS_TYPE=="mean":
                X_embeddings = tf.math.reduce_mean(X_embeddings, axis=1)
            elif self.EMBEDDINGS_TYPE=="CLS":
                X_embeddings = tf.squeeze(X_embeddings[:, 0:1, :], axis=1)
            
            inputs.append(inputs_text_ids)
            inputs.append(inputs_text_am)
            out.append(X_embeddings)
        
        if len(out)==1:
            x = out[0]
        x = tf.keras.layers.concatenate(out)

        for i in range(int(self.num_layers_feat_text)):
            x = Dense(self.num_hidden_units_feat_text, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text))(x)
            if self.batch_norm_feat_text==True:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.dropout_feat_text==True:
                x = tf.keras.layers.Dropout(self.dropout_prop_feat_text)(x)


        for i in range(num_heads):
            output = Dense(num_classes, kernel_initializer=self.initialization, kernel_regularizer=l2(self.regularization_feat_text), name="pred"+str(i+1))(x)
            outputs.append(output)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer_EMB_feat_text,                                     
                    loss=[self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif],
                    metrics=[self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif],)

    
    
    def plot_and_write_predictions(self, data, train_dataset, y_train, val_dataset, y_val, test_dataset, y_test, X_text_train=None, X_text_val=None, X_text_test=None):
        """
        Builds the one flow BERT based model for predicting the career goal variables with a regression head. 
        The model is build in a modular fashion with inout, layers etc. adjustable
        We have 3 input possibilities: text input only, numerical feature input only or input and numerical feature input

        Parameters: 
            data (object): object from Data class
            X_feat_train_list (list of np.arrays): list with t arrays containing the variables for each of the t topics

        Returns (Initializes):
            self.model: model we want to return
        """
        preds_train =  self.model.predict(train_dataset.batch(1))
        preds_val =  self.model.predict(val_dataset.batch(1))
        preds_test =  self.model.predict(test_dataset.batch(1))

        if self.MODEL=='classification':
            preds_train = np.argmax(tf.nn.softmax(preds_train,axis=2), axis=2).transpose()
            preds_val = np.argmax(tf.nn.softmax(preds_val,axis=2), axis=2).transpose()
            preds_test = np.argmax(tf.nn.softmax(preds_test,axis=2), axis=2).transpose()

        columns_pred = []
        columns_label = [] 


        for i in range(len(data.label_columns)):
            columns_pred.append('p'+str(i+1))
            columns_label.append('l'+str(i+1))

        if self.MODE=="text" or self.MODE=="text&feat":
            columns_text = ['text'+str(i+1) for i in range(len(data.text_columns))]

            #Concatenate embeddings with labels and text
            df_train = pd.concat([pd.DataFrame(preds_train, columns=columns_pred), 
                                pd.DataFrame(y_train, columns=columns_label),
                                pd.DataFrame(X_text_train, columns=columns_text)], axis=1)
            df_val = pd.concat([pd.DataFrame(preds_val, columns=columns_pred), 
                                pd.DataFrame(y_val, columns=columns_label), 
                                pd.DataFrame(X_text_val, columns=columns_text)], axis=1)
            df_test = pd.concat([pd.DataFrame(preds_test, columns=columns_pred), 
                                pd.DataFrame(y_test, columns=columns_label), 
                                pd.DataFrame(X_text_test, columns=columns_text)], axis=1)
        else:
            #Concatenate embeddings with labels and text
            df_train = pd.concat([pd.DataFrame(preds_train, columns=columns_pred), 
                                pd.DataFrame(y_train, columns=columns_label)], axis=1)
            df_val= pd.concat([pd.DataFrame(preds_val, columns=columns_pred), 
                                pd.DataFrame(y_val, columns=columns_label)], axis=1)
            df_test= pd.concat([pd.DataFrame(preds_test, columns=columns_pred), 
                                pd.DataFrame(y_test, columns=columns_label)], axis=1)

        #Save to excel file
        with pd.ExcelWriter(r'./Data/predictions_dataset.xlsx') as writer:  
            df_train.to_excel(writer, sheet_name='train')
            df_test.to_excel(writer, sheet_name='test') 
            df_val.to_excel(writer, sheet_name='val') 
        
        plot_confusion_matrix(df_test, data, "./imgs/classification/test_confusion_text_" + str(self.MODEL))
        plot_Q20_pred_label_distr(df_test, data, "./imgs/classification/test_pred_label_distr_" + str(self.MODEL))
        plot_confusion_matrix(df_train, data, "./imgs/classification/train_confusion_text_" + str(self.MODEL))
        plot_Q20_pred_label_distr(df_train, data, "./imgs/classification/train_pred_label_distr_" + str(self.MODEL))

        


    