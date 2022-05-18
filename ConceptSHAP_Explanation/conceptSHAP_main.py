"""
main function for training the concept model + 
extracting the concepts, ConceptSHAP values and completeness score
"""

import ConceptNet
import conceptSHAP_helper
import json
import sys
sys.path.append('./Classification')
from Data import Data
from Classifier import Classifier
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
import ConceptNet
import conceptSHAP_helper
from transformers import DistilBertTokenizer
from transformers import TFDistilBertModel
import matplotlib.pyplot as plt
import seaborn as sns

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
stop_word_list = dataset_variables['stop_word_list']

EMBEDDINGS_TYPE = constants['embeddings']['BERT_embeddings_type'] # Type of embedding used by the model ("CLS", "mean", "BiLSTM")
PRE_TRAINED_MODEL_NAME = constants['embeddings']['pretrained_model_name'] #BERT model we choose for contextual embedding extraction
MODEL = constants['main_params']['model'] # classification or regression head
MODE = constants['main_params']['mode'] # defines whether the input is text only, text and features or features only

n_concept = constants['ConceptSHAP_experiments']['n_concept'] # Number of concepts we want to extract
num_nearest_neighs = constants['ConceptSHAP_experiments']['num_nearest_neighs'] # Number of nearest neighbour samples we want to extract for a concept
feature_model_pretrain = constants['ConceptSHAP_experiments']['feature_model_pretrain'] #Whether the feature extraction has already been performed
concept_model_trained = constants['ConceptSHAP_experiments']['concept_model_trained'] # Whether the concept model is already trained
thres_array = constants['ConceptSHAP_experiments']['thres_array'] # Distance threshold of when a token embedding is close enough to a concept
n_size = constants['data_params']['max_sample_length'] #length of the senteces

# Path for loading the weights of the best original model
weights_filepath = './Classification/' + constants['file_paths']['best_model_weights_file_path']
    
# Initializing the model Classifier class
classifier = Classifier(constants)

# Initializing tokenizer and bert model 
tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
bert_model = TFDistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

# Initializing the Data Class
data = Data(constants, dataset_variables) 

if __name__ == '__main__':
  
    # Loading the dataset in a DataFrame
    df = pd.read_excel (r'./Data/EMS1.xlsx', sheet_name='EMS 1.0 All 7197 Items KH2 ')
    # Cleaning and spliting dataset in its label and input subcomponents 
    df, df_y, df_X_text_list, df_X_feat_list = data.prepare_X_y(df)
    
    # Splitting the labels in train, val and test
    y_train, y_test, y_val = data.split_data(df_y)
    # Splitting the text string arrays in train, val and test
    X_text_train, _, X_text_val = data.split_data(df_X_text_list[0])
    # Splitting the numerical feature input in train, val and test
    X_feat_train_list, X_feat_val_list, _, _ = data.create_X_feat_list(df_X_feat_list)


    #Creates tokens of length max_sample_length = 250 (Truncates long senteces and pads short sentences)
    # We get a dictonary of "input_ids" and "attention_mask"
    inputs_train = tokenizer(list(X_text_train), truncation=True, max_length=data.max_sample_length, padding='max_length', add_special_tokens=True, return_tensors="tf")
    input_train_ids = inputs_train["input_ids"].numpy()
    inputs_val = tokenizer(list(X_text_val), truncation=True, max_length=data.max_sample_length, padding='max_length', add_special_tokens=True, return_tensors="tf")
    input_val_ids = inputs_val["input_ids"].numpy()
    
    # get contextualized word embeddings of size N x 250 x 768 form BERT
    if not feature_model_pretrain:    
        f_train = bert_model([inputs_train["input_ids"],inputs_train["attention_mask"]])[0].numpy()
        f_val = bert_model([inputs_val["input_ids"],inputs_val["attention_mask"]])[0].numpy()

        np.save('./imdb_data/f_train_imdb.npy', f_train)
        np.save('./imdb_data/f_val_imdb.npy', f_val)

    else:
        f_train = np.load('./imdb_data/f_train_imdb.npy')
        f_val = np.load('./imdb_data/f_val_imdb.npy')

    # Initialize the classic regression model for predicting career goal from embeddings and numerical features
    # used in the first set of experiments
    classifier.build_feat_text_regression_model(
        y_train.shape[1], 
        X_feat_train_list=X_feat_train_list,
        X_emb_train_list=[np.mean(f_train, axis=1)]
        )

    # Create the datasets for the model from the label array and input lists of arrays
    # Instead of the means embeddings vector N x 768 as for the basic model the text inout is the embeddings for all word tokens 
    # of size N x 250 x 768
    X_dic_train = {"topic"+str(i+1): X_feat_train_list_i for i, X_feat_train_list_i in enumerate(X_feat_train_list)}
    X_dic_train['f_input'] = f_train
    X_dic_val = {"topic"+str(i+1): X_feat_val_list_i for i, X_feat_val_list_i in enumerate(X_feat_val_list)}
    X_dic_val['f_input'] = f_val
    train_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_dic_train, {"predictions": y_train},))
    val_dataset = tf.data.Dataset.from_tensor_slices(
                    (X_dic_val, {"predictions": y_val},))

    # Initialize or load concept model
    for count,thres in enumerate(thres_array):
        if count:
            load = './imdb_data/latest_topic_nlp.h5'
        else:
            load = False
        # Initialize or reload concept model
        topic_model_pr = ConceptNet.topic_model_nlp(classifier,
                                    X_feat_train_list,
                                    f_train,
                                    n_concept,
                                    loss1=tf.keras.losses.MeanAbsoluteError(),
                                    thres=thres,
                                    load=load)
        
        # Train concept model if not already trained
        if not concept_model_trained:
            topic_model_pr.fit(
                train_dataset.batch(classifier.batch_size_feat_text),
                validation_data=val_dataset.batch(classifier.batch_size_feat_text),
                epochs=classifier.epochs_feat_text,
                verbose=True)
    
            topic_model_pr.save_weights('./imdb_data/latest_topic_nlp.h5')
    
    # Load or save topic_vec (768 x n_concept)
    if not concept_model_trained:
        topic_vec = topic_model_pr.layers[1].get_weights()[0]
        np.save('./imdb_data/topic_vec_nlp.npy',topic_vec)
    else:
        topic_vec = np.load('./imdb_data/topic_vec_nlp.npy')

    # Normalize f_train and topic_vec
    f_train_n = f_train/(np.linalg.norm(f_train,axis=1,keepdims=True)+1e-9)
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
    
    
    # Calculte topic probability/scores (N x 250 x n_concept) for each word token
    topic_prob = np.matmul(f_train_n, topic_vec_n)
    print(topic_prob.shape)
    print('top prob')
    print(np.mean(np.max(topic_prob,axis=(0,1))))

    ########### Get concepts by 
    # extracting the top (k = num_nearest_neighs) nearest neighbour (nn) word token embeddings 
    # with the highest topic_prob

    #Instance array save the indices of the samples belonging to the concept of size n_concept x nn
    concept_instance_array = np.zeros((n_concept, num_nearest_neighs))
    #Instance array save the numerical word tokens of the samples slices belonging to the nn word tokens
    concept_nn_array = np.zeros((n_concept, num_nearest_neighs, 8))
    
    for concept in range(n_concept):
        print('----------------------- \n Concept:{}'.format(concept))

        # Get the num_nearest_neighs indices of the values with the highest topic_prob and flatten the array
        ind = np.argsort(topic_prob[:,:,concept].flatten())[::-1][:num_nearest_neighs]
        
        # Intitialize dict_count to count how often a word occurs in total in a cncepts nearest neigbour slices
        dict_count = {}

        # Iterate through the nn indices for concept c
        for jc,ind_i in enumerate(ind):
            # Get  sample instance and index within sample back from the flat ind array
            n_instance = int(np.floor(ind_i/(n_size)))
            n_index = int(ind_i-n_instance*(n_size))

            # Fill the concept_instance_array of size n_concept x nn
            concept_instance_array[concept, jc] = n_instance

            # Get the sentence sclice and the word token count for the specific word 
            temp_sentence, dict_count = conceptSHAP_helper.show_sentence(tokenizer, input_train_ids, n_instance, n_index, dict_count)
            if n_index < (len(input_train_ids[n_instance])-4) and n_index > 4:
                # Fill the concept_instance_array of size n_concept x nn x 8 with the slices consting of 8 word tokens 
                concept_nn_array[concept,jc,:] = input_train_ids[n_instance][n_index-4:n_index+4]
            #Print out the sentence slices
            print("Sentence " + str(jc) + ": " + temp_sentence)

        #Print out the word tokens that occur at least 5 times for the concept
        for key in dict_count:
            if dict_count[key]>=5 and key not in stop_word_list:
                print('Word: ' + key + " Count: " + str(dict_count[key]))
            
    np.save('./imdb_data/concept_nn_nlp.npy', concept_nn_array)
    np.save('./imdb_data/concept_index_array.npy', concept_instance_array)
    
    ##### Extracting Concept score 

    # Get predictions for original model 
    X_test = X_feat_val_list + [np.mean(f_val, axis=1)]
    classifier.model.load_weights(weights_filepath)
    true_model_pred = classifier.model.predict(X_test)
    
    # Specifiy the class we want to get the conceptSHAP values for
    spec_class =  'None' #or 0 or 1
    expl = []
    completeness_score = []

    # We compute the ConceptSHAP values for all 8 prediction heads 
    for head in range(len(data.label_columns)):
        # Get the accuracy of the original model. Our goal is to achieve a high completeness by getting close to this accuracy
        full_acc = conceptSHAP_helper.get_acc(head, true_model_pred, y_val, spec_class)
        #Baseline accuracy for a random_prediction
        null_acc = 0.5 

        # Get the n_concept Shap values for measuring the contribution of the topics
        expl_i = ConceptNet.get_shap(val_dataset, 
                                    y_val, topic_vec, topic_model_pr, 
                                    full_acc, null_acc, n_concept, 
                                    conceptSHAP_helper.get_concept_combination_acc, head, spec_class)
        expl.append(expl_i)
        print("Explanation head " + str(head) + ": " + str(expl_i))
        
        # We compute the completeness score by taking the model where all concept vectors are included (1,1,1,1).
        completeness_score_i = (conceptSHAP_helper.get_concept_combination_acc((1,1,1,1), topic_vec, val_dataset, 
                                            y_val, topic_model_pr, head, spec_class) 
                                - null_acc)/(full_acc - null_acc)
      
        completeness_score.append(completeness_score_i)
        print("Completeness score head " + str(head) + ": " + str(completeness_score_i))

    # Extract the "mean" embeddings for the samples belonging to a concept 
    # to see whether they show a similar activation pattern
    for c in range(np.shape(concept_instance_array)[0]):
        # Get indices of samples for concept c
        index = concept_instance_array[c, :].astype(int)

        # Take mean embeddings of these samples
        f_concept = pd.DataFrame(np.clip(np.mean(f_train, axis=1)[index], a_min = -1, a_max = 1))
   
        # Show embeddings in a heatmap
        plt.figure(figsize = (20, 5), facecolor = None) 
        cmap = sns.diverging_palette(240, 25, as_cmap=True)
        sns.clustermap(np.transpose(f_concept), cmap=cmap)

        plt.tight_layout(pad = 2) 
       
        # show plot
        plt.savefig("Embedding_activations_concept" + str(c) + ".png")
        plt.close()


        
    

    
    


