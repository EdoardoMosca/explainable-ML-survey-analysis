# Explaining Neural NLP Models for the Joint Analysis of Open- and Closed-Ended Survey Answers

##  1. <a name='TableofContents'></a>Table of Contents
1. [Table of Contents](#TableofContents)
2. [Overview](#Overview)
3. [Dependencies](#Dependencies)
4. [Structure and Operations](#StructureandOperations)
    
    4.1. [Classification](#Classification)
    
    4.2. [SHAP_Explanation](#SHAP_Explanation)
    
    4.3. [ConceptSHAP_Explanation](#ConceptSHAP_Explanation)
    
    4.4. [data_analysis.py](#data_analysis.py)
    
    4.5. [Data.py](#Data.py)

##  2. <a name='Overview'></a>Overview
This repo contains the code for the implemenation of the master's thesis "Explaining Neural NLP Models To Understand Students' Career Choices". 

We analyze influencing factors that drive student career decisions. Our underlying dataset is the Engineering Major Survey (EMS 1.0), collecting open and closed ended answers to questions related totopics like background and universitiy experiences.

We introduce a new approach for analyzing this survey with methods from deep learning. Moreover the approach is a general blueprint for analyzing mixed surveys containing qualitative information more efficiently and effectively.

Our approach consists of 3 parts. 

1. [Classification](#Classification): 
   
   - Prediction of students' career goals (labels) from closed-ended answers (numerical input features) and open-ended answwers (text input features). 
   - Usage of a modular neural network architecture, that combines the different input streams.  
   - Automatic convertion of qualitative text input to numerical feature input with the extraxtion of contextual text embeddings using the language model BERT.
   - Conduction of model architecture and hyperparameter tuning to achieve the best model performance. 

2. [SHAP_Explanation](#SHAP_Explanation):
    - Measurement of contextual text embedding and numerical feature contribution to the model's prediction (locally for specific predictions and globally for all predictions)
    - Measurement of text input contribution to the text embedding neurons with the highest activation (locally for specific predictions)
    - Measurement of text input contribution to the model's prediction (locally for specific predictions)
    
3. [ConceptSHAP_Explanation](#ConceptSHAP_Explanation):
    - Training of a surrogate model to extract concepts with corresponding concept vectors in an unsupervised fashion, with a "Completeness score" as a measurement for recovering the original model.
    - Characterization of latent language concepts by 
        - a) a concept vector's top k nearest neigbour word embeddings, with the sentence slice they occur in.
        - b) a word cloud consisting of the words form the sentence slices that occur most often
    - Measurement of the concepts' contribution to the model's predictions (globally for all predictions) by ConceptSHAP values.


<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->
##  3. <a name='Dependencies'></a>Dependencies
If you use conda as your virtual environment manager you can install the environment by running the following: 

    `conda env create -f mt.yml`

Otherwise the packages required can be found in ```requirements.txt```

##  4. <a name='StructureandOperation'></a>Structure and Operation

The code is structured in 3 main folders ```Classification```, ```SHAP_Explanation``` and ```ConceptSHAP_Explanation``` corresponding to the 3 experiment steps plus one extra ```Data``` folder containing all the datasets. 

In ```constants.json``` you can change all the important hyperparameters to run different experiments. The current version shows the best set of architecture variables and training parameters.

The prediction/label and input variables, we choose for our experiments from the dataset are defined in ```dataset_variables.json```.

1. Prediction variable

    **q20**: *"How likely is it that you will do each of the following in the first five years after you graduate?"*. 
        
    It provides eight career possibilities, which will be our prediction heads:

    0. **q20asbus**: Work as an employee for a small business or start-up company 
    1. **q20blbus**: Work as an employee for a medium- or large-size business
    2. **q20cnon**: Work as an employee for a non-profit organization (excluding a school or college/university)
    3. **q20dgov**: Work as an employee for the government, military, or public agency (excluding a school or college/university)
    4. **q20etch**: Work as a teacher or educational professional in a K-12 school
    5. **q20fcoll**: Work as a faculty member or educational professional in a college or university
    6. **q20hsnon**: Found or start your own for-profit organization
    7. **q20gsfor**: Found or start your own non-profit organization

    Each of them has a value range from **0** meaning **"Definitely will not"** to 4, meaning **"Definitely will"**. 
    
    However, we bin the values from 5 to 2 values (0 and 1) in order to reduce the model complexity. (Adjustet in the ```constants.json``` file) 

2. Input variables
    
    The input is split in **text variables** and **numerical feature variables**:
    
    **a) Text variables:**

    We take two different text variables from the EMS 1.0 dataset as our model input. Both of them are processed in seperate streams and merged on an embedding level.
    1. **q22career**: *We have asked a number of questions about your future plans. If you would like to elaborate on what you are planning to do, in the next five years or beyond, please do so here."*
    2. **inspire**: *"To what extent did this survey inspire you to think about your education in new or different ways? Please describe."*

    **b) Numerical feature variables:**

    We process the 119 numerical feature variables in 8 different topics as defined in ```dataset_variables.json```: 
    1. topic "learning experiences": 47 variables
    2. topic "self-efficacy": 14 variables
    3. topic "innovation outcome expectations": 4 variables
    4. topic "background characteristics": 22 variables
    5. topic "innovation interests": 7 variables
    6. topic "career goal: innovative work": 6 variables
    7. topic "job targets": 6 variables
    8. topic "current contextual influences": 13 variables

###  4.1. <a name='Classification'></a>Classification

The ```/Classification``` folder contains all files to build, train and test the survey classification model in a modular fashion as shown in the following: 


- main run files 
    1. ```train_test_text_feature_classifier_EMB.py```
    2. ```train_test_text_feature_classifier.py```
    3. ```hp_tuning_feat_text_random.py```
    4. ```hp_tuning_EMB_random.py``` 
    
    .
    
    
    Relevant parameters in ```constants.json``` for all of the 4 files: 
        
        "main_params": { #parameters for adjusting the main model architecture
            "model": # defines the the output head
                "regression",  # 8 neurons with continuous output values between 0 and 1 or 5 
                ("classification",  # 8 heads with 2 or 5 neurons each - predicting the class probability)
            "mode": # defines the model input
                "text&feat" # The model processes text and numerical features
                ("feat" # The model processes numerical features only
                /"text" # The model processes text only)
        },
        
        "data_params": { # specify experiment variations
            "data_size": false, 
            "test_size": 0.15,
            "val_size": 0.15,
            "account_bias": false, # account for individual bias within a sample 
            "standardize_features": false, # standardize within features accross samples
            "rebalance": false, # rebalance the dataset for class inbalances
            "bin": true, # bin the values for the prediction heads from 5 to 2
            "max_sample_length": 250, # specify the maximum text token sequence length
            "min_sample_length": 2, # specify the minimum text token sequence length
            "one_hot_encode_all": false, # specify whether we one hot encode all numerical features or just non-continuous ones
            "dummy_values_text": false, # insert dummy values for samples with empty strings (true) or delete them (false)
            "dummy_values_feat": true # insert dummy values for samples with nonvalid feature values (true) or delete them (false)
        }



    - training and testing the different classification model architecures  
        
        1. ```train_test_text_feature_classifier.py``` uses BERT to create contextualized text embeddings when using text variables input:

            - Relevant parameters in ```constants.json``` 
                
                    "hyper_params_feat_text_classifier": { #adjust the model architecture and training hyperparameters
                        "regularization": 0.00, 
                        "batch_norm": false, #specifies whether the model includes batch normalization
                        "dropout": true, #specifies whether the model includes dropout
                        "dropout_prop": 0.05, #specifies the dropout probability in case we use dropout
                        "num_layers": 4, #specifies the number of layers used after text and numerical feature concatenation
                        "num_pre_layers": 0, #specifies the number of layers for the different model strams of the numerical
                                            features before concatenating them with text
                        "num_hidden_units": 512, #specifies the number of hidden units per layer
                        "learning_rate":4e-4, 
                        "batch_size":2, 
                        "epochs": 100 
                    }, 

                    "embeddings": {  
                        "pretrained_model_name": "distilbert-base-uncased", #specifies the BERT model which we use from hugging face
                        "BERT_embeddings_type": # selects the way of further processing the BERT embeddings 
                            "mean" # extracts sentence meaning by computing the 768 dimensional embedding vector as the mean 
                                    over all word token embedding vectors
                            ("CLS" # extracts sentence meaning by selecting the CLS token as the 768 dimensional embedding vector
                            /"BiLSTM" # processes the word token embedding vectors further in a BiLSTM layer to 
                                        extract the sentence vector) 
                    }

            - The model weights are stored under the following path: `'Classification/tmp/checkpoint/feat_text/ --"model"-- /"` in a folder with the same timestamp. 

            - The tensorboard results will be stored under `'Classification/logs/fit/feat_text/ --"model"-- /"` in a folder with the same timestamp. 
            
                You can examine the log curves by starting tensorboard:

                    tensorboard --logdir Classification/tmp/checkpoint/feat_text/ --"model"-- 

        
        2. ```train_test_text_feature_classifier_EMB.py``` uses simple EMBedding layer to create contextualized text embeddings when using text variables input:

            - Relevant parameters in ```constants.json``` 
                
                    "hyper_params_EMB_feat_text_classifier": { #adjust the model architecture and training hyperparameters
                        "regularization": 0.00,
                        "batch_norm": false, #specifies whether the model includes batch normalization
                        "dropout": true, #specifies whether the model includes dropout
                        "dropout_prop": 0.25, #specifies the dropout probability in case we use dropout
                        "num_layers": 4, #specifies the number of layers used after text and numerical feature concatenation
                        "num_pre_layers": 0, #specifies the number of layers for the different model strams of the numerical
                                            features before concatenating them with text
                        "num_hidden_units": 512, #specifies the number of hidden units per layer
                        "learning_rate":4e-4,
                        "batch_size":4,
                        "epochs": 300,
                        "vocab_size": 999, #variety of words we can encode with tokens
                        "embbeding_dim": 8 #the size of sentence embedding vector (768 for BERT for comparison)
                    },

            - The model weights are stored under the following path: `'Classification/tmp/checkpoint/feat_text_EMB/ --"model"-- /"` in a folder with the same timestamp. 
            - The tensorboard results will be stored under `'Classification/logs/fit/feat_text_EMB/ --"model"-- /"` in a folder with the same timestamp. 

                You can examine the log curves by starting tensorboard:

                    tensorboard --logdir Classification/tmp/checkpoint/feat_text_EMB/ --"model"-- 

    - tuning the classification model with different hyperparameters: 
        
        3. ```hp_tuning_feat_text_random.py``` uses BERT to create contextualized text embeddings when using text variables input:
            - hyperparameters are defined as domains within the file
            - Relevant parameters in ```constants.json``` 
                
                    "embeddings": {  
                        "pretrained_model_name": "distilbert-base-uncased", #specifies the BERT model which we use from hugging face
                        "BERT_embeddings_type": # selects the way of further processing the BERT embeddings 
                            "mean" # extracts sentence meaning by computing the 768 dimensional embedding vector as the mean 
                                    over all word token embedding vectors
                            ("CLS" # extracts sentence meaning by selecting the CLS token as the 768 dimensional embedding vector
                            /"BiLSTM" # processes the word token embedding vectors further in a BiLSTM layer to 
                                        extract the sentence vector) 
                    }

            - The model weights are stored under the following path: `'Classification/tmp/checkpoint/feat_text_hp/ --"model"-- /"` in a folder with the same timestamp. 
            - The tensorboard results will be stored under `'Classification/logs/hparam_tuning/feat_text_random/ --"model"-- /"` in a folder with the same timestamp. 
            
                You can examine the log curves by starting tensorboard:

                    tensorboard --logdir Classification/tmp/checkpoint/feat_text_hp/ --"model"-- 

        4. ``hp_tuning_EMB_random.py`` uses simple EMBedding layer to create contextualized text embeddings when using text variables input:
            - hyperparameters are defined as domains within the file
            - The model weights are stored under the following path: `'Classification/tmp/checkpoint/feat_text_EMB_hp/ --"model"-- /"` in a folder with the same timestamp. 
            - The tensorboard results will be stored under `'Classification/logs/hparam_tuning/EMB_random/ --"model"-- /"` in a folder with the same timestamp. 
            
                You can examine the log curves by starting tensorboard:

                    tensorboard --logdir Classification/tmp/checkpoint/feat_text_EMB_hp/ --"model"-- 
    

- class files: 
    Helper class to run the main files
    1.  ```Data.py```: Data class enclosing all functions related to the data preprocessing and dataset creation
    2. ```Classifier.py```: Classifier class enclosing all functions related to model creation for the different model variations

- helper file ```embeddings.py```: Helper file to get the BERT embeddings and the tokens for the EMBedding layer


###  4.2. <a name='SHAP_Explanation'></a>SHAP_Explanation

The ```/SHAP_Explanation``` folder contains all files to run 3 explanation experiments based on SHAP. 
It explains the classification model with the best performance, having the following architecture:
- Input: Text (q22career) + numerical features (all of the 8 topics)
- Prediction head: regression
- Extraction of text embeddings: BERT + "mean" over word vectors
- Number of layers for festure topic streams before text and numerical feature concatenation: 0
- Number of layers after text and numerical feature concatenation: 4
- Number of hidden units per layer: 512
- Batch norm: no
- Dropout: No

Main jupyter notebooks: 

Calculating the SHAP values, which measure and visualize the feature contribution to the model's predictions:
- ```main_text_feature_classifier_split_model.ipynb``` 
    1. Experiment: calculates and visualizes SHAP values measuring the contribution of text embedding and numerical feature input to the model's 8 prediction heads (locally for specific samples and globally for all samples)
    2. Experiment: calculates and visualizes SHAP values measuring the contribution of word tokens to the text embedding neurons with the highest activation (extracted from the experimenmnt) locally for a specific sample
    
    The weights for the best model are loaded from ```constants.json```:
        
            "file_paths": {
                "best_model_weights_file_path": "tmp/checkpoint/feat_text/regression/20210913-115927/"
            },
        

- ```main_text_feature_classifier_one_model.ipynb``` 
    
    3. Experiment: calculates and visualizes SHAP values measuring the contribution of word tokens  to the model's 8 prediction heads locally for a specific sample

        For this experiment we implement a new version of the model processing the text input to the prediction in one flow by including BERT (Impelemnted in ```Classifier.py```). 

        - The model weights are stored under the following path: `'../Classification/tmp/checkpoint/feat_text/ --"model"-- /"` in a folder with the same timestamp. 
        
        - The weights for this model are loaded from ```constants.json```:
            
                "file_paths": {
                    "best_one_flow_model_weights_file_path": "tmp/checkpoint/feat_text/regression/20210928-101701/"
                }, 

All SHAP plots are stored in ```imgs/SHAP_explanation``` folder.

SHAP specific parameters in ```constants.json``` for both files: 
        
        "SHAP_experiments": {
            "sample_list": [151, 157, 119, 183], #samples for local explanations
            "sample": 151, #single sample for explanation
            "experiment": "SHAP_explain" # specifies whether we want to "train" or explain the one_flow model
            },


###  4.3. <a name='ConceptSHAP_Explanation'></a>ConceptSHAP_Explanation
The ```/ConceptSHAP_Explanation``` folder contains all files to run the ConceptSHAP experiments:

- main run file ```ConceptSHAP```: 
    1. trains the surrogate model for extracting the concept vectors 
    2. extracts the concepts by calculating the nearest word token embedding neighbours to a concept vector
        - showing sentence slices containing the top k word tokens 
        - showing word clouds with the most frequently occuring words from all top k sentence slices for one topic
    3. calculates the concept's contribution to the model's prediction by getting the SHAP values for the concept vectors
    4. calculating the "completeness score", meaning the surrogates model ability to recover the original prediction
    5. calculating the text sentence embedding activations for samples belonging to a specific concept to check for the concept's coherency

    ConceptSHAP specific parameters in ```constants.json```: 
            
            "ConceptSHAP_experiments": {
                "n_concept": 4, #number of concepts we want to extract
                "num_nearest_neighs": 100, #number of nearest neighbour word tokens we want to attribute to a concept
                "feature_model_pretrain": true, #whether the embedding features have already been extracted
                "concept_model_trained": true, #whether the concept model is already pretrained
                "thres_array": [0.1] #threshold for considering a distance between the concept vector and the word token embedding as close enough. If below we ignore the word token.
            },


- model file ```ConceptNet.json```: concept model file implementing all functions related to concept model creation and SHAP value extraction
- helper file ```constants.json```: helper file for calculating accuracies and extracting the sentence sclices


###  5.4. <a name='data_analysis.py'></a>data_analysis.py
#File for creating all plots analyzing the data and the model's predictions.

###  5.5. <a name='Data'></a>Data

Stores all the datasets. 

- `EMS 1.0.xlsx`: The original EMS 1.0 dataset we use for all the experiments
- `predictions_dataset.xlsx`: Stores the predictions and the corresponding labels and text strings, in case the input includes text variables.

