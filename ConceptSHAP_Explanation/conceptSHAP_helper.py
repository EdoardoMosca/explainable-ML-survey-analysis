"Helper file for conceptSHAP_main.py"

import numpy as np
import copy

def get_acc(head, prediction, y_true, spec_class = "None"):
    """
    Computes the accuracy for a specific prediction head, either for both classes or for one of them

    Parameters: 
        head(int): which of the 8 prediction heads to look at
        prediction (np.array): array of size N x 8 for model predictions
        y_true (np.array): array of size N x 8 for model ground truth labels
        spec_class (int / string): 0, 1 or "None" for the class, we want to compute the accuracy for

    Returns:
        acc (float): Accuracy
    """
    #if we want to compute the accuracy for a specific class
    if spec_class != "None": 
        true_num = 0
        class_num = 0
        #if we want to compute the accuracy/precision for a specific class
        for x in range(np.shape(y_true)[0]):
            #check whether the label is of the class we wanted 
            if (y_true[x, head] == spec_class):
                class_num +=1
                #check whether the label and the prediction are equal
                if (np.round(prediction[x,head])==y_true[x,head]): 
                    true_num += 0
                    
        acc = true_num/class_num
    #if we want to compute the accuracy for both classes
    else: 
        #count samples, where label and the prediction are equal and devide by total sample size
        acc = np.sum(np.round(prediction[:,head])==y_true[:,head])*1.0/np.shape(prediction)[0]
    return acc

def get_concept_combination_acc(binary_sample, topic_vec, val_dataset, y_val, topic_model_pr, head, spec_class="None"):
    """
    Computes the accuracy for a specific modification of concept vectors

    Parameters: 
        binary_sample (tuple): Tuple of shape (c1, c2, .. ci) with ci = 0/1 indicating whether the concept is considered or not
        topic_vec (np.array): The topic vesctors of shape (768, n_concept)
        val_dataset: Validation dataset
        y_va (np.array)l: validation label of size N x 8
        topic_model_pr: Concept Model
        head(int): which of the 8 prediction heads to look at
        spec_class (bool / string): 0, 1 or "None" for the class, we want to compute the accuracy for

    Returns:
        acc (float): Accuracy
    """
    #Modify concept vectors: Set topic vectors in the concept_model to zero dependent on the binary_sample tuple
    topic_vec_temp = copy.copy(topic_vec) 
    topic_vec_temp[:,np.array(binary_sample)==0] = 0
    topic_model_pr.layers[1].set_weights([topic_vec_temp])

    #Predict with modified concept input
    prediction = topic_model_pr.predict(val_dataset.batch(1))

    #Return accuracy for the prediction of the modified model
    return get_acc(head, prediction, y_val, spec_class)


def show_sentence(tokenizer, input_ids, n_instance, n_index, dict_count):
    """
    Show the sentence for the word embeddings closest to a certain concept vector

    Parameters: 
        tokenizer: the tokenizer from huggingface we used for tokenizing the strings 
        input_ids: the ids corresponding to the strings as extracted by the tokenizer of size N x 250
        n_instance (int): the sample containing the closest word token
        n_index (int): the word token which is closest to the concept vector
        dict_count (dict): empty dictionary to count how often a word occurs

    Returns:
        ss:
        dict_count (dict): dictionary to count how often a word occurs
    """

    # Get a slice of 8 tokens for the sample instance. 
    # We take 4 tokens on the left and 3 tokens to the right of the actual word token.
    # If the word token is too close to an edge we take less tokens
    x_temp = input_ids[n_instance]
    if n_index < (len(x_temp)-4) and n_index > 4:
        slice = x_temp[n_index-4:n_index+4]
        
    elif n_index >= (len(x_temp)-4):
        slice = x_temp[n_index-4:]

    elif n_index <= 4:
        slice = x_temp[:n_index+4]
    
    # Get the sentence corresponding to the tokens
    sentence_slice = tokenizer.decode(slice)
    
    # Count the number a word token occurs in the sentence slice
    for id in slice:
        if tokenizer.decode(int(id)) not in dict_count:
            dict_count[tokenizer.decode(int(id))] = 1

        else:
            dict_count[tokenizer.decode(int(id))] += 1

    return sentence_slice, dict_count
