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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_feat_text, 
                                    self.num_hidden_units_feat_text, self.batch_norm_feat_text, 
                                    self.dropout_feat_text, self.dropout_prop_feat_text, self.regularization_feat_text)
        
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
        
        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream with BERT embeddings as inputs
            if self.EMBEDDINGS_TYPE=="BiLSTM":
                # Stream for BiLSTM with 3 dim text input
                for i in range(len(X_emb_train_list)):
                    # 3 dim input
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1], X_emb_train_list[i].shape[2]), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    # BiLSTM layer
                    x = Bidirectional(LSTM(512))(inputs_text)
                    out.append(x)
            else:
                # Stream for mean and CLS embedding types with 2 dim text input
                for i in range(len(X_emb_train_list)):
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1],), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    out.append(inputs_text)
        
        # Concatenate all streams and processes model to regression output
        # Compile model
        self.up_stream_regression(out, num_heads, 
                                inputs, self.num_layers_feat_text, self.num_hidden_units_feat_text, 
                                self.batch_norm_feat_text, self.dropout_feat_text, self.dropout_prop_feat_text, 
                                self.regularization_feat_text, self.optimizer_feat_text)


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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_feat_text, 
                                    self.num_hidden_units_feat_text, self.batch_norm_feat_text, 
                                    self.dropout_feat_text, self.dropout_prop_feat_text, self.regularization_feat_text)
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
       
        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream with BERT embeddings as inputs
            if self.EMBEDDINGS_TYPE=="BiLSTM":
                # Stream for BiLSTM with 3 dim text input
                for i in range(len(X_emb_train_list)):
                    # 3 dim input
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1], X_emb_train_list[i].shape[2]), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    # BiLSTM layer
                    x = Bidirectional(LSTM(512))(inputs_text)
                    out.append(x)
            else:
                for i in range(len(X_emb_train_list)):
                    # Stream for mean and CLS embedding types with 2 dim text input
                    inputs_text = tf.keras.layers.Input(shape=(X_emb_train_list[i].shape[1],), name="embeddings"+str(i+1))
                    inputs.append(inputs_text)
                    out.append(inputs_text)
        
        # Concatenate all streams and processes model to classification output
        # Compile model
        self.up_stream_classification(out, outputs, num_heads, num_classes, 
                                    inputs, self.num_layers_feat_text, self.num_hidden_units_feat_text, 
                                    self.batch_norm_feat_text, self.dropout_feat_text, self.dropout_prop_feat_text, 
                                    self.regularization_feat_text, self.optimizer_feat_text)
    
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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_EMB_feat_text, 
                                    self.num_hidden_units_EMB_feat_text, self.batch_norm_EMB_feat_text, 
                                    self.dropout_EMB_feat_text, self.dropout_prop_EMB_feat_text, self.regularization_EMB_feat_text)
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()
        
        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream with token input and extra embedding layer
            for i in range(len(X_token_train_list)):
                inputs_text = tf.keras.layers.Input(shape=(X_token_train_list[i].shape[1],), name="embeddings"+str(i+1))
                # Embedding Layer - size N x X_token_train_list[i].shape[1] x self.embedding_dim_EMB
                x = tf.keras.layers.Embedding(self.vocab_size_EMB, self.embedding_dim_EMB, input_length=X_token_train_list[i].shape[1])(inputs_text)
                # Embedding Layer - size N x self.embedding_dim_EMB
                x = tf.keras.layers.GlobalMaxPooling1D()(x)
                
                inputs.append(inputs_text)
                out.append(x)
        
        # Concatenate all streams and processes model to regression output
        # Compile model
        self.up_stream_regression(out, num_heads, 
                                inputs, self.num_layers_EMB_feat_text, self.num_hidden_units_EMB_feat_text, 
                                self.batch_norm_EMB_feat_text, self.dropout_EMB_feat_text, self.dropout_prop_EMB_feat_text, 
                                self.regularization_EMB_feat_text, self.optimizer_EMB_feat_text)

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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_EMB_feat_text, 
                                    self.num_hidden_units_EMB_feat_text, self.batch_norm_EMB_feat_text, 
                                    self.dropout_EMB_feat_text, self.dropout_prop_EMB_feat_text, self.regularization_EMB_feat_text)
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy() 
        
        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream with token input and extra embedding layer
            for i in range(len(X_token_train_list)):
                inputs_text = tf.keras.layers.Input(shape=(X_token_train_list[i].shape[1],), name="embeddings"+str(i+1))
                # Embedding Layer - size N x X_token_train_list[i].shape[1] x self.embedding_dim_EMB
                x = tf.keras.layers.Embedding(self.vocab_size_EMB, self.embedding_dim_EMB, input_length=X_token_train_list[i].shape[1])(inputs_text)
                # Embedding Layer - size N x self.embedding_dim_EMB
                x = tf.keras.layers.GlobalMaxPooling1D()(x)
                
                inputs.append(inputs_text)
                out.append(x)
        
        # Concatenate all streams and processes model to classification output
        # Compile model
        self.up_stream_classification(out, outputs,num_heads, num_classes, 
                                    inputs, self.num_layers_EMB_feat_text, self.num_hidden_units_EMB_feat_text, 
                                    self.batch_norm_EMB_feat_text, self.dropout_EMB_feat_text, self.dropout_prop_EMB_feat_text, 
                                    self.regularization_EMB_feat_text, self.optimizer_EMB_feat_text)                                  
                   
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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_feat_text, 
                                    self.num_hidden_units_feat_text, self.batch_norm_feat_text, 
                                    self.dropout_feat_text, self.dropout_prop_feat_text, self.regularization_feat_text)

        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy() 
        
        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream input tokens and BERT model included
            # Get input for ids and mask
            inputs_text_ids = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_ids", dtype='int32')
            inputs_text_am = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_am", dtype='int32')                       
            # Initialize BERT model with fixed weights
            bert_model = TFDistilBertModel.from_pretrained(pre_trained_model_name)
            bert_model.trainable = False
            # forward pass through BERT: size N x 250 x 768
            X_embeddings = bert_model([inputs_text_ids, inputs_text_am])[0]

            # Reduce dimesnion to N x 768
            if self.EMBEDDINGS_TYPE=="mean":
                X_embeddings = tf.math.reduce_mean(X_embeddings, axis=1)
            elif self.EMBEDDINGS_TYPE=="CLS":
                X_embeddings = tf.squeeze(X_embeddings[:, 0:1, :], axis=1)
            
            inputs.append(inputs_text_ids)
            inputs.append(inputs_text_am)
            out.append(X_embeddings)

        # Concatenate all streams and processes model to regression output
        # Compile model
        self.up_stream_regression(out, num_heads, 
                                inputs, self.num_layers_feat_text, self.num_hidden_units_feat_text, 
                                self.batch_norm_feat_text, self.dropout_feat_text, self.dropout_prop_feat_text, 
                                self.regularization_feat_text, self.optimizer_feat_text)

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
            #Compute output for feature stream 
            out, inputs = self.feat_streams(X_feat_train_list, self.num_pre_layers_feat_text, 
                                    self.num_hidden_units_feat_text, self.batch_norm_feat_text, 
                                    self.dropout_feat_text, self.dropout_prop_feat_text, self.regularization_feat_text)
            
        elif self.MODE=="text":
            #No numerical feature inputs if just text
            inputs = []
            out = inputs.copy()

        if self.MODE=="text&feat" or self.MODE=="text":
            #Compute output for text stream input tokens and BERT model included
            # Get input for ids and mask
            inputs_text_ids = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_ids", dtype='int32')
            inputs_text_am = tf.keras.layers.Input(shape=(data.max_sample_length,), name="text_am", dtype='int32')                       
            # Initialize BERT model with fixed weights
            bert_model = TFDistilBertModel.from_pretrained(pre_trained_model_name)
            bert_model.trainable = False
            # forward pass through BERT: size N x 250 x 768
            X_embeddings = bert_model([inputs_text_ids, inputs_text_am])[0]

            # Reduce dimesnion to N x 768
            if self.EMBEDDINGS_TYPE=="mean":
                X_embeddings = tf.math.reduce_mean(X_embeddings, axis=1)
            elif self.EMBEDDINGS_TYPE=="CLS":
                X_embeddings = tf.squeeze(X_embeddings[:, 0:1, :], axis=1)
            
            inputs.append(inputs_text_ids)
            inputs.append(inputs_text_am)
            out.append(X_embeddings)
        
        # Concatenate all streams and processes model to classification output
        # Compile model
        self.up_stream_classification(out, outputs, num_heads, num_classes, 
                                    inputs, self.num_layers_feat_text, self.num_hidden_units_feat_text, 
                                    self.batch_norm_feat_text, self.dropout_feat_text, self.dropout_prop_feat_text, 
                                    self.regularization_feat_text, self.optimizer_feat_text)

    def plot_and_write_predictions(self, data, train_dataset, y_train, val_dataset, y_val, test_dataset, y_test, X_text_train=None, X_text_val=None, X_text_test=None):
        """
        Makes predictions with trained model, writes predictions in excel file and plots confusion matrices + distributions
 
        Parameters: 
            data (object): object from Data class
            train_dataset: training dataset for model input 
            y_train (np.array): train label array of size: N x 8
            val_dataset: val dataset for model input
            y_val (np.array): val label array of size: N x 8
            test_dataset: test dataset for model input
            y_test (np.array): test label array of size: N x 8
            X_text_train (np.array): train array with text strings (N_train x num_text_variables)
            X_text_val (np.array): val array with text strings (N_val x num_text_variables) 
            X_text_test (np.array): test array with text strings (N_test x num_text_variables)
        """
        # Make model predictions
        preds_train =  self.model.predict(train_dataset.batch(1))
        preds_val =  self.model.predict(val_dataset.batch(1))
        preds_test =  self.model.predict(test_dataset.batch(1))

        # Get the class with the highest prediction probability
        if self.MODEL=='classification':
            preds_train = np.argmax(tf.nn.softmax(preds_train,axis=2), axis=2).transpose()
            preds_val = np.argmax(tf.nn.softmax(preds_val,axis=2), axis=2).transpose()
            preds_test = np.argmax(tf.nn.softmax(preds_test,axis=2), axis=2).transpose()

        # Get column names for numerical features and labels
        columns_pred = []
        columns_label = [] 
        for i in range(len(data.label_columns)):
            columns_pred.append('p'+str(i+1))
            columns_label.append('l'+str(i+1))
        
        
        if self.MODE=="text" or self.MODE=="text&feat":
            # Get column names for text
            columns_text = ['text'+str(i+1) for i in range(len(data.text_columns))]

            #Concatenate predictions with labels and text
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
            #Concatenate predictions with labels
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
        
        #Plot confusion matrix and label distributions for train, val and test data
        plot_confusion_matrix(df_test, data, "./imgs/classification/test_confusion_text_" + str(self.MODEL))
        plot_Q20_pred_label_distr(df_test, data, "./imgs/classification/test_pred_label_distr_" + str(self.MODEL))
        plot_confusion_matrix(df_train, data, "./imgs/classification/train_confusion_text_" + str(self.MODEL))
        plot_Q20_pred_label_distr(df_train, data, "./imgs/classification/train_pred_label_distr_" + str(self.MODEL))

        
    def up_stream_regression(self, out, num_heads, inputs, num_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization, optimizer):
        """
        Computes the model output given a list of outputs from downstream layers. 
        Creates and compiles the model 
 
        Parameters: 
            out (list of keras modules): list of outputs form the downstream layers (combine text and numerical features)
            num_heads (int): number of prediction heads
            inputs (list of keras inputs): inputs for the model (combine text and numerical features)
            num_layers (int): number of upstream layers for the model (deoth)
            num_hidden_units (int): number of neurons for a layer (width)
            batch_norm (bool): wehther a layer should contain batch_norm
            dropout (bool): wehther a layer should contain dropout
            dropout_prop (float): dropout probability in case droput is used
            regularization (float): regularization for the model
            optimizer (bool): which optimizer we use
        """

        # if the out list is one x is just its element
        if len(out)==1:
            x = out[0]
        # otherwise concatenate all streams
        else: 
            x = tf.keras.layers.concatenate(out)

        # we stack several layers of relus
        x = self.fc_relu_loop(x, num_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization)

        # The regression output has 8 neurons with a continuous prediction each representing one head.
        outputs = Dense(num_heads, kernel_initializer=self.initialization, kernel_regularizer=l2(regularization), name="predictions")(x)
        
        # Initialize and compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer,                                     
                    loss=self.loss_reg,
                    metrics=self.metrics_reg)

    def up_stream_classification(self, out, outputs,num_heads, num_classes, inputs, num_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization, optimizer):
        """
        Computes the model output given a list of outputs from downstream layers. 
        Creates and compiles the model 
 
        Parameters: 
            out (list of keras modules): list of outputs form the downstream layers (combine text and numerical features)
            num_heads (int): number of prediction heads
            num_classes (int): number of classes per prediction head
            inputs (list of keras inputs): inputs for the model (combine text and numerical features)
            num_layers (int): number of upstream layers for the model (deoth)
            num_hidden_units (int): number of neurons for a layer (width)
            batch_norm (bool): wehther a layer should contain batch_norm
            dropout (bool): wehther a layer should contain dropout
            dropout_prop (float): dropout probability in case droput is used
            regularization (float): regularization for the model
            optimizer (bool): which optimizer we use
        """
        
        # if the out list is one x is just its element
        if len(out)==1:
            x = out[0]
        # otherwise concatenate all streams
        else: 
            x = tf.keras.layers.concatenate(out)

        # we stack several layers of relus
        x = self.fc_relu_loop(x, num_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization)

        # The regression output has 8 heads each with a probability prediction for num_class neurons for each class 
        for i in range(num_heads):
            output = Dense(num_classes, kernel_initializer=self.initialization, kernel_regularizer=l2(regularization), name="pred"+str(i+1))(x)
            outputs.append(output)

        # Initialize and compile model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer,                                     
                    loss=[self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif, self.loss_classif],
                    metrics=[self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif, self.metrics_classif],)

    def fc_relu_loop(self, x, num_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization):
        """
        Computes several forward passes thorugh a stack of fully connected relu layers.
 
        Parameters: 
            x (keras output): output from the downstream layers
            num_layers (int): number of upstream layers for the model (deoth)
            num_hidden_units (int): number of neurons for a layer (width)
            batch_norm (bool): wehther a layer should contain batch_norm
            dropout (bool): wehther a layer should contain dropout
            dropout_prop (float): dropout probability in case droput is used
            regularization (float): regularization for the model
        """
        for i in range(int(num_layers)):
            x = Dense(num_hidden_units, activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l2(regularization))(x)
            if batch_norm==True:
                # Add batch_norm
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout==True:
                # Add dropout
                x = tf.keras.layers.Dropout(dropout_prop)(x)
        return x

    def feat_streams(self, X_feat_train_list, num_pre_layers, num_hidden_units, batch_norm, dropout, dropout_prop, regularization):
        """
        Initializes the inputs for the features and further creates a model with a stack of fully conncected layer streams 
        for each topic
 
        Parameters: 
            X_feat_train_list (list of np.arrays): list with 8 numerical feature arrays for each topic
            num_pre_layers (int): number of layers for each topic stream before concatenation of topics 
            num_hidden_units (int): number of neurons for a layer (width)
            batch_norm (bool): wehther a layer should contain batch_norm
            dropout (bool): wehther a layer should contain dropout
            dropout_prop (float): dropout probability in case droput is used
            regularization (float): regularization for the model
        """
        #Initialize input with numerical features
        inputs = [tf.keras.layers.Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
        
        #If num_pre_layers_feat_text > 0 we process the numerical features in different streams depending on the topic they belong to
        if int(num_pre_layers) > 0: 
            x_i_list = []
            # Each topic gets proessed independently
            for i in range(len(X_feat_train_list)):
                #First layer with dropout and batch_norm optional
                x_i = Dense(int(num_hidden_units/len(X_feat_train_list)), activation=tf.nn.relu, kernel_initializer=self.initialization, kernel_regularizer=l1(regularization))(inputs[i])
                if batch_norm==True:
                    x_i = tf.keras.layers.BatchNormalization()(x_i)
                if dropout==True:
                    x_i= tf.keras.layers.Dropout(dropout_prop)(x_i)
                
                #Following layer with dropout and batch_norm optional
                x_i = self.fc_relu_loop(x_i, num_pre_layers, int(num_hidden_units/len(X_feat_train_list)), batch_norm, dropout, dropout_prop, regularization)
            
                #Creating a list of stream outputs which we concatenate later
                x_i_list.append(x_i) 
                out = x_i_list.copy()
        else:
                #Just copy inputs if we directly take the numerical features 
                # without processing them in seperate streams
                out = inputs.copy()
        return out, inputs