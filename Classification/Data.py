"""
Data class enclosing all functions related to the data preprocessing and dataset creation
"""
import sys
sys.path.append('./.')
#from data_analysis import plot_Q20_statistics
import pandas as pd
import numpy as np
import copy
from sklearn import preprocessing
from embeddings import run_bert
import tensorflow as tf

class Data():
    def __init__(self, constants, dataset_variables):
        """
        Initializes the Data class
        """
        self.MODE = constants['main_params']['mode'] #defines whether the input is text only, text and features or features only
        self.MODEL = constants['main_params']['model'] #defines whether our model has a regression or classification head
        
        if self.MODE=="text":
            #only text input 
            self.feature_columns = {} #no feature variables
            self.text_columns = dataset_variables['text_columns'] #text variable list
         
        elif self.MODE=="feat":
            #only numerical feature input
            self.text_columns = [] #no text variables
            self.feature_columns = dataset_variables['feature_columns'] #dictionary of topic - numerical variable list, key - value pairs. 
        
        elif self.MODE=="text&feat":
            #text and numerical feature input 
            self.feature_columns = dataset_variables['feature_columns'] #dictionary of topic - numerical variable list, key - value pairs. 
            self.text_columns = dataset_variables['text_columns'] #text variable list
        
        self.label_columns = dataset_variables['label_columns'] #prediction head variable list
        self.one_hot_feat = dataset_variables['one_hot_feat'] #list of numerical features that must be one-hot encoded
        
        self.data_dict = constants['data_params'] #dictionary with data related variables
        self.one_hot_encode_all = self.data_dict['one_hot_encode_all'] #If "true" all numerical features are one_hot encoded
        self.data_size = self.data_dict['data_size'] #If not false we take a subset of the dataset   
        self.bin = self.data_dict['bin'] #If "true", we bin the prediction variables form 5 to 2 values
        self.account_bias = self.data_dict['account_bias'] #If "true", we normalize values accross features within samples
        self.standardize_features = self.data_dict['standardize_features'] #If true we normalize values accross samples within features
        self.scaler = preprocessing.StandardScaler() # Choice of scaler for standardization of features
        self.rebalance = self.data_dict['rebalance']  #If "true" we delete samples to create an equal class distribution
        self.test_share = self.data_dict['test_size'] #Defines the share of the test dataset
        self.val_share = self.data_dict['val_size'] #Defines the share of the validation dataset
        self.max_sample_length = self.data_dict['max_sample_length'] #Defines the maximum length of the text samples by # of tokens
        self.min_sample_length = self.data_dict['min_sample_length'] #Defines the minimum length of the text samples by # of tokens
        
        self.dummy_values_feat = self.data_dict['dummy_values_feat'] #If "true" we insert dummy values (-1) for empty feature values, if "false" we drop the sample
        self.dummy_values_text = self.data_dict['dummy_values_text'] #If "true" we insert an empty string ('') for empty text values, if "false" we drop the sample
    

    def prepare_raw_data(self, df):
        """
        Cleans and prepares (variable selection, one_hot_encoding, imputation) the raw dataset

        Parameters: 
            df(DataFrame): raw dataset input

        Returns:
            df(DataFrame): cleaned and prepared dataset
        """

        # defines the subset of features we want to select from the dataset in a list 
        total_features = []
        for topic_features in self.feature_columns.values():
            total_features += topic_features

        subset_features = self.label_columns + self.text_columns + total_features

        # selects the subset of features
        df = pd.DataFrame(df, columns = subset_features)

        # If we insert no dummy values, delet samples with any empty value
        if self.dummy_values_feat==False and self.dummy_values_text==False:
            df = df.dropna()

        print(f'full data size: {len(df)}')

        #Drop samples which have a missing value for label variables
        for l in self.label_columns:
            for x in df.index:
                if df.loc[x, l]==(-9) or df.loc[x, l]==(-7):
                    df.drop(x, inplace = True)
        
        print(f'reduced data size after dropping of missing label samples: {len(df)}')
        
        # If we process text input
        if len(self.text_columns) != 0: 
            for column in self.text_columns:
                # insert an empty string for empty samples with 'nan'
                df[column] = df[column].replace(np.nan, '', regex=True)

                if self.dummy_values_text==False:
                    # drop samples with the text variables being empty/ 
                    # having a string_value size below min_sample_length
                    for x in df.index:
                        if (df.loc[x, column].count(' ') + 1) < self.min_sample_length:
                            df.drop(x, inplace = True)
        print(f'reduced data size after dropping of missing text samples: {len(df)}')
        
        # If we process numerical feature input
        if len(self.feature_columns) != 0: 
            for topic_features in self.feature_columns.values():
                
                if self.dummy_values_feat==True:
                    # insert a constant dummy value of -1 for samples with unregular/undefined values
                    df[topic_features[:]] = df[topic_features[:]].replace([-9, -8, -7, -5, -4], -1)

                else:
                    # drop samples with unregular/undefined values
                    for f in topic_features:
                        for x in df.index:
                            if df.loc[x, f]==(-9) or df.loc[x, f]==(-8) or df.loc[x, f]==(-7) or df.loc[x, f]==(-5) or df.loc[x, f]==(-4):
                                df.drop(x, inplace = True)
        
            #Process for one_hot encoding of numerical features
            if self.one_hot_encode_all==True:
                #if we one_hot encode all variables our one_hot_feat dictionary just becomes the regular numerical feature dictionary
                one_hot_feat = copy.deepcopy(self.feature_columns)
            else:
                #Get the one_hot_feat dictionary defined in the dataset_variables.json
                one_hot_feat = self.one_hot_feat
            
            # Iterates through the topic keys, each with a list of features
            for topic, one_hot_feat_topic_i in one_hot_feat.items():
                # Iterates thorugh each feature in the list of features belonging to a topic
                for feat in one_hot_feat_topic_i: 
                    # Returns a DataFrame with variable columns for each value of the original variable, each being either 0 or 1
                    df_one_hot = pd.get_dummies(df[feat], prefix=feat)
                    # Drop the column representing the dummy value -1 as we represent the "empty" case with all new variables being 0
                    if self.dummy_values_feat==True:
                        df_one_hot = df_one_hot.drop([col for col in df_one_hot.columns if '-1' in col], axis=1)
                    # Drop column feat as it is now represented by one_hot variable columns
                    df = df.drop(feat, axis = 1)
                    # Join the encoded df_one_hot to the original df
                    df = df.join(df_one_hot)
                    #Add one_hot encoded feature names to feature_columns
                    self.feature_columns[topic].remove(feat) 
                    self.feature_columns[topic] += list(df_one_hot.columns)
            
            print(f'reduced data size after dropping of missing feature samples: {len(df)}')
        
        #Take a subset of the dataset if data_size is specified
        if self.data_size!=False: 
            df = df[1000:(1000+self.data_size)]

        return df
        

    def standardize_individuals(self, label_df):
        """
        # normalize values accross label values within samples to account for bias in the label answers

        Parameters: 
            label_df(DataFrame): dataset with label columns before normalization

        Returns:
            label_df(DataFrame): dataset with label columns after normalization
        """

        for i in range(label_df.shape[0]): 
            # Take each sample individually
            df = label_df.iloc[i]

            # Compute mean and standard deviation for the label values of each sample after centering it around zero
            df = df - 2
            mean = df.mean()
            std = df.std()
            
            # Shift and rescale the values 
            if std == 0:
                std = 1
            df = np.rint(((df - mean)/std) + 2)

            # Replace old samplewith normalized sample
            label_df.iloc[i]=df
        return label_df


    def split_data(self, df):
        """
        Split dataset in train, validation and test data subsets

        Parameters: 
            df(DataFrame): total dataset before splitting

        Returns:
            train, test, val (np.array): subsets
        """

        #Take subsets based on their share
        df_test = df.iloc[:(int(self.test_share*len(df)))]
        df_val = df.iloc[(int(self.test_share*len(df))): (int((self.test_share+self.val_share)*len(df)))]
        df_train = df.iloc[(int((self.test_share+self.val_share)*len(df))):]
        
        #Convert DataFrame to np.arrays
        train = np.asarray(df_train)
        test = np.asarray(df_test)
        val = np.asarray(df_val)
        return train, test, val


    def prepare_X_y(self, df):
        """
        Takes the raw dataset, cleans it and returns its preprocessed label, 
        feature and text subcomponents

        Parameters: 
            df(DataFrame): raw dataset before splitting

        Returns:
            df(DataFrame): Cleaned and preprocessed DataFrame
            labels (DataFrame): DataFrame only containing the 8 label variables
            X_text (DataFrame): DataFrame only containing the 2 text variables
            X_feature_list (list of DataFrames): List of 8 DataFrames each with variables for the 8 topics
        """
        # Clean and preprocess raw dataset
        df = self.prepare_raw_data(df)

        # Subselect label DataFrame
        df_labels = df.loc[:, self.label_columns[:]]

        # Standardize labels to account for individual bias
        if self.account_bias==True:
            df_labels = self.standardize_individuals(df_labels)
        
        # Bin labels from 5 values to 2 for an easier prediction
        if self.bin==True:
            df_labels = self.bin_labels(df_labels)
        
        # Subselect text DataFrame
        if len(self.text_columns) != 0:
            #Creates list of text variables
            df_X_text_list = [df[column] for column in self.text_columns]

        # Create list of subselected numerical feature DataFrames
        if len(self.feature_columns) != 0: 
            df_X_feature_list = [df.loc[:, topic_features[:]] for topic_features in self.feature_columns.values()]

        return df, df_labels, df_X_text_list, df_X_feature_list

    def bin_labels(self, df_label):
        """
        Bins the label values from 5 values to 2

        Parameters: 
            df_label(DataFrame): dataset with label columns before binning

        Returns:
            df_label(DataFrame): dataset with label columns after binning
        """

        #Perform the binning for individually each of the columns 
        for j in range(df_label.shape[1]):

            #Compute the median for all sample values of one column
            column_med = df_label.iloc[:,j].median()
            #We choose a different binning strategy based on whether we rebalance the classes later or not
            if self.rebalance==False:  
                # If the median for a column is below 2 we assign all values above 2 to label 1 
                # and all values equal or lower to 0
                if column_med<2:
                    for i in range(df_label.shape[0]):
                        if df_label.iloc[i,j] > column_med:
                            df_label.iloc[i,j] = 1
                        elif df_label.iloc[i,j] <= column_med:
                            df_label.iloc[i,j] = 0
                # If the median for a column is equal or higher than 2 we assign all values above or higher than 2 
                # to label 1 and all lower to 0
                elif column_med>=2:
                    for i in range(df_label.shape[0]):
                        if df_label.iloc[i,j] >= column_med:
                            df_label.iloc[i,j] = 1
                        elif df_label.iloc[i,j] < column_med:
                            df_label.iloc[i,j] = 0

                #Different bining strategy without dependency on the median
                # for i in range(df_label.shape[0]):
                #     if df_label.iloc[i,j] >= 2:
                #         df_label.iloc[i,j] = 1
                #     elif df_label.iloc[i,j] < 2:
                #         df_label.iloc[i,j] = 0

            if self.rebalance==True:
                # If the value is above or equal to the median it gets assigned to one and 0 if lower
                for i in range(df_label.shape[0]):
                    if df_label.iloc[i,j] >= column_med:
                        df_label.iloc[i,j] = 1
                    elif df_label.iloc[i,j] < column_med:
                        df_label.iloc[i,j] = 0

        return df_label

    def rebalance_f(self, X_text_train, X_emb_train_list, X_feat_train_list, y_train):
        """
        Rebalances the dataset components to get a balanced distribution for the 2 (binned) classes. 

        Parameters: 
            X_text_train(np.array): list of text strings for text variables
            X_emb_train_list(list of np.array): list of text embedding arrays
            X_feat_train_list(list of np.array): list of numerical feature arrays
            y_train(np.array): prediction label array

        Returns:
            X_text_train(np.array): reduced list of text strings for text variables
            X_emb_train_list(list of np.array): reduced list of text embedding arrays
            X_feat_train_list(list of np.array): reduced list of numerical feature arrays
            y_train(np.array): balanced prediction label array
        """

        #Delete samples as long as not as many samples with label 0 and label 1
        while all([np.count_nonzero(y_train[:,i] == 0) != np.count_nonzero(y_train[:,i] == 1) for i in range(y_train.shape[1])]):
            #Get indices where all labels for the 8 heads are 0
            index, = np.where(np.sum(y_train, axis=1)==0) 
            if len(index) == 0:
                break

            #Delete the first sample in the indices list for all subcomponents X_text_train, X_emb_train_list, X_feat_train_list, y_train
            y_train = np.delete(y_train, index[0], 0)
            if self.MODE=="text&feat" or self.MODE=="text":
                for i, X_emb_train in enumerate(X_emb_train_list):
                    X_emb_train_list[i] = np.delete(X_emb_train, index[0], 0)
                X_text_train = np.delete(X_text_train, index[0], 0)
            if self.MODE=="text&feat" or self.MODE=="feat":
                for i, X_feat_train in enumerate(X_feat_train_list):
                    X_feat_train_list[i] = np.delete(X_feat_train, index[0], 0) 
        #Print data distribution
        #plot_Q20_statistics(pd.DataFrame(y_train), self, 'hist', name='bin_label_stats_post_balancing')

        return X_text_train, X_emb_train_list, X_feat_train_list, y_train

    
    def create_X_emb_list(self, df, df_X_text_list, df_y, EMBEDDINGS_TYPE, tokenizer, bert_model):
        """
        Creates a list containing the contextual embedding arrays for the text variables
        for each training, validation, and test

        Parameters: 
            df (DataFrame): Cleaned and preprocessed dataset 
            df_X_text_list (DataFrame): List of DataFrames each for one text variable
            df_y (DataFrame): Cleaned and preprocessed label sub dataset 
            EMBEDDINGS_TYPE (str): type of embedding used by the model ("CLS", "mean", "BiLSTM")
            tokenizer: tokenizer, which we use to convert word strings into tokens
            bert_model: distill bert language model to extract contextual word embeddings

        Returns:
            X_emb_train_list (list of np.array): list of arrays (N_train x 768) with contextual text embeddings for each of the text variables
            X_emb_val_list (list of np.array): list of arrays (N_val x 768) with contextual text embeddings for each of the text variables
            X_emb_test_list (list of np.array): list of arrays (N_test x 768) with contextual text embeddings for each of the text variables
        """

        #df_X_emb_list = []
        X_emb_train_list = []
        X_emb_val_list = []
        X_emb_test_list = []
        for df_X_text in df_X_text_list:
            # for each text variable compute the contextual embeddings, which we add then to a list for all the text variables
            X_emb = run_bert(list(df_X_text), self.max_sample_length, EMBEDDINGS_TYPE, tokenizer, bert_model)
            # if we train a BiLSTM on top of the embeddings X_emb is 3 dim: N x 250 x 768, which we add to the list 
            if EMBEDDINGS_TYPE=="BiLSTM":
                # create 3 dim np.arrays of size: N (subset) x 250 x 768
                y_train, y_test, y_val = self.split_data(df_y)
                X_emb_train = np.zeros([y_train.shape[0], X_emb.shape[1], X_emb.shape[2]])
                X_emb_val = np.zeros([y_val.shape[0], X_emb.shape[1], X_emb.shape[2]])
                X_emb_test = np.zeros([y_test.shape[0],X_emb.shape[1], X_emb.shape[2]])
                #We iterate through the 250 tokens and split all the N x 768 dim DataFrames in test, val and train
                for token_i in range(X_emb.shape[1]):
                    df_X_emb_pos = pd.DataFrame(X_emb[:, token_i, :], index=df.index)
                    X_emb_train_pos, X_emb_test_pos, X_emb_val_pos = self.split_data(df_X_emb_pos)
                    # Standardize the arrays accross samples
                    if self.standardize_features == True:
                        X_emb_train_pos = self.scaler.fit_transform(X_emb_train_pos)
                        X_emb_test_pos = self.scaler.fit_transform(X_emb_test_pos)
                        X_emb_val_pos = self.scaler.fit_transform(X_emb_val_pos)
                    # Add the split 2 dim token arrays to the 3 dim arrays
                    X_emb_train[:, token_i, :] = X_emb_train_pos
                    X_emb_val[:, token_i, :] = X_emb_val_pos
                    X_emb_test[:, token_i, :] = X_emb_test_pos

            # if the embeddings type is "CLS" or "mean" X_emb is 2 dim: N x 768, which we add to the list
            else:
                X_emb_train, X_emb_test, X_emb_val = self.split_data(pd.DataFrame(X_emb, index=df.index))
                #df_X_emb_list.append(pd.DataFrame(X_emb, index=df.index))
                if self.standardize_features == True:
                    X_emb_train = self.scaler.fit_transform(X_emb_train)
                    X_emb_test = self.scaler.fit_transform(X_emb_test)
                    X_emb_val = self.scaler.fit_transform(X_emb_val)

            #Add the arrays for each text variable to the overall emb_list
            X_emb_train_list.append(X_emb_train.copy())
            X_emb_val_list.append(X_emb_val.copy())
            X_emb_test_list.append(X_emb_test.copy())
            return X_emb_train_list, X_emb_val_list, X_emb_test_list
        

    def create_X_feat_list(self, df_X_feat_list):
        """
        Creates a list containing the arrays for the numerical feature variables
        for each training, validation, and test

        Parameters: 
            df_X_feat_list (list of DataFrame): list containing 8 DataFrames with the variables belonging to the 8 topics 

        Returns:
            X_feat_train_list (list of np.array): list containing 8 arrays of size: (N_train x #topic_variables)
            X_feat_val_list (list of np.array): list containing 8 arrays of size: (N_val x #topic_variables)
            X_feat_test_list (list of np.array): list containing 8 arrays of size: (N_test x #topic_variables)
            feat_name_list (list of strings): list containing all numerical feature variable names
        """

        X_feat_train_list = []
        X_feat_val_list = []
        X_feat_test_list = []
        feat_name_list = []

        # Going through each list element (Topic) and splitting each topic DataFrame in train, val and test subset
        for df_X_feat in df_X_feat_list:
            # Appending topic variable names to the overall feature_name_list
            feat_name_list.append(df_X_feat.columns.tolist())
            # Splitting DataFrames and receiving np.arrays
            X_feat_train, X_feat_test, X_feat_val = self.split_data(df_X_feat)
            # Standardize each np.array across samples
            if self.standardize_features == True:
                X_feat_train = self.scaler.fit_transform(X_feat_train)
                X_feat_test = self.scaler.fit_transform(X_feat_test)
                X_feat_val = self.scaler.fit_transform(X_feat_val)
            # Append topic array again to a list containing an array for each topic
            X_feat_train_list.append(X_feat_train.copy())
            X_feat_val_list.append(X_feat_val.copy())
            X_feat_test_list.append(X_feat_test.copy())  
        
        return X_feat_train_list, X_feat_val_list, X_feat_test_list, feat_name_list

    def create_X_dic(self, X_emb_list, X_feat_list):
        """
        Creates a dictionary with keys for unique idetification of model input

        Parameters: 
            X_emb_list (list of np.array): list of arrays (N x 768) with contextual text embeddings for each of the text variables
            X_feat_list (list of np.array): list containing 8 arrays of size: (N x #topic_variables)
           
        Returns:
            X_dic (dic): dictionary of topics and embeddings
        """
        if self.MODE=="feat" or self.MODE=="text&feat":
            #Attach all feat list elements to the dictionary as a value beonging to a topic key
            X_dic = {"topic"+str(i+1): X_feat_list_i for i, X_feat_list_i in enumerate(X_feat_list)}

        if self.MODE=="text&feat":
            #Attach all text embedding list elements to the dictionary as a value beonging to a topic key
            for i, X_emb_list_i in enumerate(X_emb_list):
                X_dic['embeddings'+str(i+1)] = X_emb_list_i
        
        elif self.MODE =="text":
            #Create a dictionary with just the text embedding list elements
            X_dic = {"embeddings"+str(i+1): X_emb_list_i for i, X_emb_list_i in enumerate(X_emb_list)}
        
        return X_dic

    def create_tf_dataset(self, X_emb_list, X_feat_list, y):

        """
        Creates a dataset for model training from the dictionaries

        Parameters: 
            X_emb_list (list of np.array): list of arrays (N x 768) with contextual text embeddings for each of the text variables
            X_feat_list (list of np.array): list containing 8 arrays of size: (N x #topic_variables)
            y (np.array): data_label array of size N x 8
           
        Returns:
            dataset (tf.data.Dataset): dictionary of topics and embeddings
        """

        #First create the dictionary for unique input identification
        X_dic = self.create_X_dic(X_emb_list, X_feat_list)

        if self.MODEL=='regression':
            #Create tf Dataset from dictionary with y being one output head for the regression model
            dataset = tf.data.Dataset.from_tensor_slices(
                        (X_dic, {"predictions": y},))
        elif self.MODEL=='classification':
            #Create tf Dataset from dictionary with y being split into 8 output head for the classification model
            y_dic = {"pred"+str(i+1): y[:,i] for i in range(y.shape[1])}
            
            dataset = tf.data.Dataset.from_tensor_slices(
                (X_dic, y_dic,))

        return dataset

        