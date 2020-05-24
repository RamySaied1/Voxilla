import sys
import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 
#from paramaters import *
MFCC_Size=40
def normalize(v):
    return v/tf.sqrt(tf.reduce_sum(v**2, axis=-1, keep_dims=True)+1e-6)

class Model():
    def __init__(self,args,mode):
        self.numHidden=args.LSTMHiddenUnits
        self.numLayers=args.LSTMLayers
        self.numProjection=args.ProjectionUnits
        self.mode=mode
        if mode=='train':
            self.speakersPerBatch=args.NumSpeakers
            self.utterancesPerBatch=args.NumUtterances
            #optimizer
            self.optimizer=tf.train.GradientDescentOptimizer(args.learningRate)
        self.prepareModel()
    def prepareModel(self):
        #The Architecture is 3 LSTM layers every layer consists on 768 hidden units every layer is followed
        #by a projection layer of size 256 units
        with tf.variable_scope("lstm"):
            self.batch= tf.placeholder(dtype=tf.float64, shape=[None, self.speakersPerBatch*self.utterancesPerBatch,5* MFCC_Size], name="input_batch")
            #List of size numLayers each element is a LSTM layer of 768 units followed by a projection layer
            lstm_cells=[rnn.LSTMCell(num_units=self.numHidden, num_proj=self.numProjection) for i in range(self.numLayers)]
            #Our network
            self.lstm=rnn.MultiRNNCell(lstm_cells) 
            #self.embeddingVector=tf.placeholder(dtype=tf.float64,shape=[256],name="embeddingVector")
            output,_=tf.nn.dynamic_rnn(cell=self.lstm,inputs=self.batch,dtype=tf.float64)
            self.embeddingVector=output[-1]
            self.embeddingVector=normalize(self.embeddingVector)
        if self.mode=="train":
            with tf.variable_scope("sim_matrix"):
                self.calculateSimilarityMatrix()
            with tf.variable_scope("loss"):
                self.calculateLoss()
            with tf.variable_scope("optimize"):
                self.optimizeOperation()


    
    #Similarity matrix is of size (Num_speakers*Num_utterances,Num_speakers) 
    #embeddings shape is (Num_speakers*Num_utterances,NUM_Projection)
    def calculateSimilarityMatrix(self):
        self.w = tf.get_variable("w", initializer= np.array([10], dtype=np.float64))
        self.b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float64))
        self.similarityMatrix=tf.zeros(shape=[self.speakersPerBatch*self.utterancesPerBatch,self.speakersPerBatch])
        reshapedEmbeddings=tf.reshape(self.embeddingVector,shape=[self.speakersPerBatch,self.utterancesPerBatch,self.numProjection])
        #there are two types of centroids one per speaker and other per embedding vector
        centroidsPerSpeaker=tf.reduce_mean(reshapedEmbeddings, axis=1)
        centroidsPerEmbedding=(tf.reshape(tf.reduce_sum(reshapedEmbeddings,axis=1, keepdims=True)-reshapedEmbeddings,shape=[self.speakersPerBatch   *self.utterancesPerBatch,self.numProjection]))/(self.utterancesPerBatch-1)

        utteranceNumber=0
        self.similarityMatrix = tf.concat(
            [tf.concat([tf.reduce_sum(centroidsPerEmbedding[i*self.utterancesPerBatch:(i+1)*self.utterancesPerBatch,:]*reshapedEmbeddings[j,:,:], axis=1, keep_dims=True) if i==j
                        else tf.reduce_sum(centroidsPerSpeaker[i:(i+1),:]*reshapedEmbeddings[j,:,:], axis=1, keep_dims=True) for i in range(self.speakersPerBatch)],
                       axis=1) for j in range(self.speakersPerBatch)], axis=0)
        self.similarityMatrix = tf.abs(self.w)*self.similarityMatrix+self.b 
    def calculateLoss(self):
        target=np.repeat(np.arange(self.speakersPerBatch), self.utterancesPerBatch)
        self.totalLoss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.similarityMatrix, labels=target))
        #tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.similarityMatrix, labels=target)
        
    def optimizeOperation(self):
        gradients=self.optimizer.compute_gradients(self.totalLoss)
        self.optimize = self.optimizer.apply_gradients(gradients)