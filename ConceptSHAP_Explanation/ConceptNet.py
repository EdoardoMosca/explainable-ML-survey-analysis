"""
ConceptNET enclosing all functions related to Concept model creation and SHAP value extraction
"""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow.keras as k

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import numpy as np
from numpy import inf
from numpy.random import seed
from scipy.special import comb
import tensorflow as tf
seed(0)
tf.random.set_seed(0)
# global variables
init = k.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)


class Weight(Layer):
  """Simple Weight class."""

  def __init__(self, dim, **kwargs):
    self.dim = dim
    super(Weight, self).__init__(**kwargs)

  def build(self, input_shape):
    # creates a trainable weight variable for this layer.
    self.kernel = self.add_weight(
        name='proj', shape=self.dim, initializer=init, trainable=True)
    super(Weight, self).build(input_shape)

  def call(self, x):
    return self.kernel


def topic_loss_nlp(loss1):
  """creates loss for topic model"""
  def loss_f(y_true, y_pred):
      loss = tf.reduce_mean(input_tensor=loss1(y_true, y_pred))
      return loss
  return loss_f

def topic_model_nlp(predict, X_feat_train_list, f_train, n_concept, loss1, thres, load=False):
    """
    Show the sentence for the word embeddings closest to a certain concept vector

    Parameters: 
        predict (keras model): the (second part) prediction model for classification with embeddings and numerical features as input
        X_feat_train_list (list of np.array): list of 8 array with numerical feature variables for each of the 8 topics
        f_train (np.array): the contextual word embeddings of size N x 250 x 768
        n_concept (int): number of concepts we want to derive
        loss1: basic loss for our loss function
        thresh (float): threshold for the distance of a word embedding to a concept vector
        load (bool): defines whether we load an old model

    Returns:
        topic_model_pr: returns the surrogate concept model training the concept vectors 
    """
  
    # input consisting of numerical features belonging to 8 topics
    inputs = [Input(shape=(X_feat_train_list[i].shape[1],), name="topic"+str(i+1)) for i in range(len(X_feat_train_list))]
    out = inputs.copy()
    # f_input of size (None, 250, 768)
    f_input = Input(shape=(f_train.shape[1],f_train.shape[2]), name='f_input')
    # overall input combines numerical features and f_input 
    inputs.append(f_input)
    # normalizing f_input
    f_input_n =  Lambda(lambda x:K.l2_normalize(x,axis=(2)))(f_input)

    # trainable topic vector of size (768, n_concept)
    topic_vector = Weight((f_train.shape[2], n_concept))(f_input)
    topic_vector_n = Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)

    # topic_prob calculated as the scalar product of f_input and the topic_vector
    # size: batchsize * 250 * n_concept
    topic_prob = Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector])
    #Thresholding and normalizing topic_probability 
    topic_prob_n = Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
    topic_prob_mask = Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
    topic_prob_am = Lambda(lambda x:x[0]*x[1])([topic_prob, topic_prob_mask])
    topic_prob_sum = Lambda(lambda x: K.sum(x, axis=2, keepdims=True)+1e-3)(topic_prob_am)
    topic_prob_nn = Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
    
    # trainable rec_vector_1 (size: n_concept * 500) and rec_vector_2 (size: 500 x 768)
    rec_vector_1 = Weight((n_concept, 500))(f_input)
    rec_vector_2 = Weight((500, f_train.shape[2]))(f_input)
    # rec_layer_1 size: batchsize * 250 * 500
    rec_layer_1 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
    # rec_layer_2 size: batchsize * 250 * 768
    rec_layer_2 = Lambda(lambda x:K.relu(K.dot(x[0],x[1])))([rec_layer_1, rec_vector_2])
    # We choose the embedding_type as "mean": batchsize * 768
    rec_layer_f2 = Lambda(lambda x:K.mean(x, axis=1))(rec_layer_2)

    # We append and concatenate the recovered embeddings (batchsize * 768) again to the numerical features
    # We now feed this as x again in our original classification prediction model
    out.append(rec_layer_f2)
    x = tf.keras.layers.concatenate(out)

    #Same structure as for Classification "regression" model
    for i in range(int(predict.num_layers_feat_text)):
        x = Dense(predict.num_hidden_units_feat_text, activation=tf.nn.relu, kernel_initializer=predict.initialization, kernel_regularizer=l2(predict.regularization_feat_text))(x)
    pred = Dense(8, kernel_initializer=predict.initialization, kernel_regularizer=l2(predict.regularization_feat_text), name="predictions")(x)

    #Create the model
    topic_model_pr = tf.keras.models.Model(inputs=inputs, outputs=pred)
    #topic_model_pr.layers[-1].trainable = True
    #topic_model_pr.layers[1].trainable = False

    # Add regularization term to loss to ensure closeness of samples within a topic and spatial distance
    topic_model_pr.add_loss(-0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(tf.transpose(tf.reshape(topic_prob_n,(-1,n_concept))),k=16,sorted=True).values))
        +0.1*tf.reduce_mean(input_tensor=(tf.matmul(tf.transpose(topic_vector_n), topic_vector_n) - tf.eye(n_concept)))
        )
    
    # Add metric to measure comprehensability
    topic_model_pr.add_metric(tf.reduce_mean(
        input_tensor=tf.nn.top_k(K.transpose(K.reshape(topic_prob_n,(-1,n_concept))),k=16,sorted=True).values), 
        name='comprehensability')
    
    # Compile model
    topic_model_pr.compile(
        loss=topic_loss_nlp(loss1),
        optimizer=predict.optimizer_feat_text, 
        metrics=predict.metrics_reg,
        run_eagerly=True,
        )

    print(topic_model_pr.summary())
    if load:
        topic_model_pr.load_weights(load)
    return topic_model_pr


def shap_kernel(n, k):
    """Returns kernel of shapley in KernelSHAP."""
    return (n-1)*1.0/((n-k)*k*comb(n, k))


def get_shap(val_dataset, y_val, topic_vec, model_shap, full_acc, null_acc, n_concept, get_acc_f, head, spec_class):
    """
    Returns ConceptSHAP value explanation.

    Parameters: 
        nc: 
        val_dataset: list of 8 array with numerical feature variables for each of the 8 topics
        y_val (np.array): array for validation labels of size: N x 8
        topic_vec (np.array): The topic vesctors of shape (768, n_concept)
        model_shap: concept model
        full_acc (float): best accuracy of original model
        null_acc (float): random accuracy
        n_concept (int): number of concepts
        get_acc_f (f): function to calculate the accuracy of a given prediction and label data array
        head (int): specifies the prediction head
        spec_class (string/bool): specified which class we want to focus on (either 0, 1, "none" (meaning both))

    Returns:
        expl (tuple): returns n_concept values showing the contribution of the concepts to the specified class predictions
    """
    # create all different combinations of tuples with size n_concept consiting of 0 and 1s.
    # This is the occlusion of the concept vectors 
    inputs = list(itertools.product([0, 1], repeat=n_concept))

    # Calculate the completeness (based on the accuracy) of the model for all these different concept vector occlusions 
    # (similar to normal SHAP with the input features)
    outputs = [(get_acc_f(k, topic_vec, val_dataset, y_val, model_shap, head, spec_class)-null_acc)/
                (full_acc-null_acc) for k in inputs]
    #Calculate SHAP kernel based on the different accuracies for all occlusion combinations
    kernel = [shap_kernel(n_concept, np.sum(ii)) for ii in inputs]
    x = np.array(inputs)
    y = np.array(outputs)
    k = np.array(kernel)
    k[k == inf] = 10000
    xkx = np.matmul(np.matmul(x.transpose(), np.diag(k)), x)
    xky = np.matmul(np.matmul(x.transpose(), np.diag(k)), y)
    expl = np.matmul(np.linalg.pinv(xkx), xky)
    return expl