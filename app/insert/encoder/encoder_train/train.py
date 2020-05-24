from model import Model
from load_dataset import readDataset
import os
import argparse
from paramaters import *
import tensorflow as tf
from generate_batches import generateBatch
import numpy as np
def train():
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--TotalSpeakers",type=int,required=True, help="total Number of speakers in dataset")
    
    parser.add_argument("--LSTMLayers",type=int,required=False,default=3, help="Number of LSTM layers ")
    parser.add_argument("--LSTMHiddenUnits",type=int,required=False,default=768, help="Number of Hidden units in LSTM Layers ")
    parser.add_argument("--ProjectionUnits",type=int,required=False,default=256, help="Number of projection units ")
    
    
    parser.add_argument("--NumSpeakers",type=int,required=False,default=64, help="Number of speakers in a batch")
    parser.add_argument("--NumUtterances",type=int,required=False,default=10, help="Number of utterances of a speaker in a batch")
    
    parser.add_argument("--datasetPath",type=str,required=True,help="Path containing The Processed dataset")
    parser.add_argument("--outputPath",type=str,required=True,help="Path to saving checkpoints and model ")

    parser.add_argument("--numSteps",type=int,required=True,help="Total Number of training steps")
    parser.add_argument("--learningRate",type=float, required=False,default=0.01,help="Learning Rate")

    args = parser.parse_args()

    model=Model(args)
    with tf.Session() as sess:
        print(f"[train.py]: Initializing Global Variables")
        initializationVariables=tf.global_variables_initializer()
        writer = tf.summary.FileWriter(args.outputPath, sess.graph)
        checkpointPath = tf.train.get_checkpoint_state(args.outputPath+"/checkpoints")
        sess.run(initializationVariables)
        saver = tf.train.Saver()
        for stepNumber in range(0,args.numSteps):
            inputBatch=generateBatch(args.NumSpeakers,args.NumUtterances,args.TotalSpeakers,args.datasetPath)
            inputBatch=np.asarray(inputBatch)
            inputBatch=np.reshape(inputBatch,(1,4,200))
            #embeddedVector=sess.run([model.embeddingVector],feed_dict={model.batch:inputBatch})
            #embeddedVector=np.reshape(embeddedVector,(4,256))
            #similarityMatrix=sess.run([model.similarityMatrix],feed_dict={model.embeddingVector:embeddedVector})
            #similarityMatrix=np.reshape(similarityMatrix,(4,2))
            embeddedVector,_,loss,_=sess.run([model.embeddingVector,model.similarityMatrix,model.totalLoss,model.optimize],feed_dict={model.batch:inputBatch})
            #loss=np.asarray(loss)
            #loss=np.reshape(loss,(4,))
            print(type(model.totalLoss),type(loss),model.totalLoss,loss)
            #_=sess.run([model.optimize],feed_dict={model.totalLoss:loss})
            print("model.w",model.w,"model.b",model.b)
            print("Loss:",str(loss),"\n Embedding Vector: ",embeddedVector)
            #writer.add_summary(loss, stepNumber)
            if stepNumber % 10 ==0:
                save_path = saver.save(sess, args.outputPath+"/model.ckpt")
                print("model saved in file: %s / %d th step" % (save_path, stepNumber))


   
train()