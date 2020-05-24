import sys
import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 
from generate_batches import generateBatch
#from paramaters import *
MFCC_Size=40
def normalize(v):
    return v/tf.sqrt(tf.reduce_sum(v**2, axis=-1, keep_dims=True)+1e-6)

class Model():
    def __init__(self,args):
        self.numHidden=args.LSTMHiddenUnits
        self.numLayers=args.LSTMLayers
        self.numProjection=args.ProjectionUnits
        self.batch=tf.placeholder(dtype=tf.float64, shape=[None, args.NumSpeakers*args.NumUtterances,5* MFCC_Size], name="input_batch")
        #Learnable paramaters,GE2E suggests w=10,b=-5 as initial value
        self.w = tf.get_variable("w", initializer= np.array([10], dtype=np.float64))
        self.b = tf.get_variable("b", initializer= np.array([-5], dtype=np.float64))
        self.embeddingVector=tf.Variable(tf.zeros([args.NumSpeakers*args.NumUtterances,args.ProjectionUnits]))
        self.totalLoss=tf.Variable(tf.zeros([1]))
        #placeholder(dtype=tf.float64,shape=[256],name="embeddingVector")
        #batch size
        self.speakersPerBatch=args.NumSpeakers
        self.utterancesPerBatch=args.NumUtterances
        self.totalSpeakers=args.TotalSpeakers
        self.datasetPath=args.datasetPath
        #optimizer
        self.optimizer=tf.train.GradientDescentOptimizer(0.01)
        self.createArch()
    def createArch(self):
        #The Architecture is 3 LSTM layers every layer consists on 768 hidden units every layer is followed
        #by a projection layer of size 256 units
        
        with tf.variable_scope("lstm"):
            self.batch= tf.convert_to_tensor(generateBatch(self.speakersPerBatch,self.utterancesPerBatch,self.totalSpeakers,self.datasetPath))
            #List of size numLayers each element is a LSTM layer of 768 units followed by a projection layer
            lstm_cells=[rnn.LSTMCell(num_units=self.numHidden, num_proj=self.numProjection) for i in range(self.numLayers)]
            #Our network
            self.lstm=rnn.MultiRNNCell(lstm_cells) 
            #self.embeddingVector=tf.placeholder(dtype=tf.float64,shape=[256],name="embeddingVector")
            output,_=tf.nn.dynamic_rnn(cell=self.lstm,inputs=self.batch,dtype=tf.float64)
            self.embeddingVector=output[-1]
            self.embeddingVector=normalize(self.embeddingVector)
            self.similarityMatrix=tf.zeros(shape=[self.speakersPerBatch*self.utterancesPerBatch,self.speakersPerBatch])
            
            #output,_=tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float64, time_major=True)  
    '''def forwardPropagation(self):
        with tf.variable_scope("lstm"):
            self.batch=generateBatch(self.speakersPerBatch,self.utterancesPerBatch,self.totalSpeakers,self.datasetPath)
            #This line unrolls all the inputs and returns all the outputs of the RNN network
            output,_=tf.nn.dynamic_rnn(cell=self.lstm,inputs=self.batch,dtype=tf.float64,time_major=True)
            #d-vector is the output of the last layer
            self.embeddingVector=output[-1]
            #normalize d-vector according to the paper
            self.embeddingVector=normalize(self.embeddingVector)
            print(f"EmbeddingVector {self.embeddingVector}\n")
            return self.embeddingVector
    '''

    
    #Similarity matrix is of size (Num_speakers*Num_utterances,Num_speakers) 
    #embeddings shape is (Num_speakers*Num_utterances,NUM_Projection)
    def calculateSimilarityMatrix(self):
        with tf.variable_scope("sim_matrix"):
            reshapedEmbeddings=tf.reshape(self.embeddingVector,shape=[self.speakersPerBatch,self.utterancesPerBatch,self.numProjection])
            #there are two types of centroids one per speaker and other per embedding vector
            centroidsPerSpeaker=tf.reduce_mean(reshapedEmbeddings, axis=1)
            centroidsPerEmbedding=(tf.reshape(tf.reduce_sum(reshapedEmbeddings,axis=1, keepdims=True)-reshapedEmbeddings,shape=[64*10,self.numProjection]))/(self.utterancesPerBatch-1)

            utteranceNumber=0
            for row in self.similarityMatrix:
                for cell_number in range(0,len(row)):
                    if utteranceNumber%self.utterancesPerBatch==cell_number:
                        #This utterance was said by current speaker
                        #add centroid per embedding
                        self.similarityMatrix[row][cell_number]=centroidsPerEmbedding[utteranceNumber]
                        utteranceNumber+=1
                    else:
                        #add centroid per speaker
                        self.similarityMatrix[row][cell_number]=centroidsPerSpeaker[utteranceNumber]
            self.similarityMatrix = tf.abs(self.w)*self.similarityMatrix+self.b 
    def calculateLoss(self):
        with tf.variable_scope("loss"):
            target=tf.convert_to_tensor(np.repeat(np.arange(self.speakersPerBatch), self.utterancesPerBatch))
            self.totalLoss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.similarityMatrix, labels=target))
    