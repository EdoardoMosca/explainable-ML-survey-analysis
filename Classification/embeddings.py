"""
helper file to get tokens or contextualized embeddings by running BERT
"""
import numpy as np
import tensorflow as tf

def get_tokens(X, vocab_size, max_length):
    """
    Converts words of text strings into numerical tokens

    Parameters: 
        X (Dataframe): Dataframe with text strings for one of the text variables
        vocab_size (int): vocabulary size (#of tokens being able to describe different word)
        max_length (int): length of the token sequence

    Returns:
        tokens: dataset with label columns after binning
    """
    #Get token numbers for each word in a sentence
    encoded_docs = [tf.keras.preprocessing.text.one_hot(x, vocab_size) for x in list(X)]
    #Ensure each sentence has the same length with padding shorter sentences
    tokens = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    return tokens

def run_bert(X, max_sample_length, embeddings_type, tokenizer, bert_model):
    """
    Creates the contextual word embeddings by running the text string input thorugh the BERT model

    Parameters: 
        X (list of strings): text input for one text variable
        max_sample_length (int): length of one token sequence/sentence
        embeddings_type (string): type of embedding used by the model ("CLS", "mean", "BiLSTM")

    Returns:
        X_embeddings (np.array): Contextual word embeddings of size: N x 768 ("mean" / "CLS") and N x 250 x 768 ("BiLSTM")
    """

    #Creates tokens of length max_sample_length = 250 (Truncates long senteces and pads short sentences)
    # We get a dictonary of "input_ids" and "attention_mask"
    inputs = tokenizer(X, truncation=True, max_length=max_sample_length, padding='max_length', add_special_tokens=True, return_tensors="tf")
    
    #Print the first 10 samples with their masks and tokens
    # for ids in inputs["input_ids"][:10]:
    #     print(tokenizer.decode(ids))
    # print(inputs)

    # Get contextual embeddings (N x 250 x 768) by forward pass of tokens through BERT model
    X_embeddings = bert_model([inputs["input_ids"],inputs["attention_mask"]])

    if embeddings_type=="mean":
        # Calculate the mean of all 250 word tokens for each (of the 768) embeddings dimensions to get a N x 768 sentence array
        return np.mean(X_embeddings[0].numpy(), axis=1)
    elif embeddings_type=="CLS":
        # Take the first CLS token array to get a N x 768 sentence array
        return X_embeddings[0][:,0,:].numpy()
    elif embeddings_type=="BiLSTM":
        # Pass the N x 250 x 768 array
        return X_embeddings[0].numpy()
