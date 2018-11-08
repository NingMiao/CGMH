from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import reader
from config import config
config=config()
from tensorflow.python.client import device_lib
import os 
os.environ['CUDA_VISIBLE_DEVICES']=config.GPU

from utils import *

logging = tf.logging

def data_type():
  return tf.float16 if config.use_fp16 else tf.float32

class PTBModel(object):
  #The language model.

  def __init__(self, is_training, is_test_LM=False):
    self._is_training = is_training
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    self._input=tf.placeholder(shape=[None, config.num_steps], dtype=tf.int32)
    self._target=tf.placeholder(shape=[None, config.num_steps], dtype=tf.int32)
    self._sequence_length=tf.placeholder(shape=[None], dtype=tf.int32)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input)
    softmax_w = tf.get_variable(
          "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)
    output = self._build_rnn_graph(inputs, self._sequence_length, is_training)

    output=tf.reshape(output, [-1, config.hidden_size])
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
      # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [-1, self.num_steps, vocab_size])
    self._output_prob=tf.nn.softmax(logits)
      # Use the contrib sequence loss and average over the batches
    mask=tf.sequence_mask(lengths=self._sequence_length, maxlen=self.num_steps, dtype=data_type())
    loss = tf.contrib.seq2seq.sequence_loss(
      logits,
      self._target,
      mask, 
      average_across_timesteps=True,
      average_across_batch=True)

    # Update the cost
    self._cost = loss


    #self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer()
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

  def _build_rnn_graph(self, inputs, sequence_length, is_training):
    return self._build_rnn_graph_lstm(inputs, sequence_length, is_training)

  def _get_lstm_cell(self, is_training):
    return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, sequence_length, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell( is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
    outputs, states=tf.nn.dynamic_rnn(cell=cell, inputs=inputs, sequence_length=sequence_length, dtype=data_type())

    return outputs
  


def run_epoch(sess, model, input, sequence_length, target=None, mode='train'):
  #Runs the model on the given data.
  if mode=='train':
    #train language model
    _,cost = sess.run([model.train_op, model.cost], feed_dict={model.input: input, model.target:target, model.sequence_length:sequence_length})
    return cost
  elif mode=='test':
    #test language model
    cost = sess.run(model.cost, feed_dict={model.input: input, model.target:target, model.sequence_length:sequence_length})
    return cost
  else:
    #use the language model to calculate sentence probability
    output_prob = sess.run(model.output_prob, feed_dict={model.input: input, model.sequence_length:sequence_length})
    return output_prob

def main(_):
  if os.path.exists(config.forward_log_path) and config.mode=='forward':
    os.system('rm '+config.forward_log_path)
  if os.path.exists(config.backward_log_path) and config.mode=='backward':
    os.system('rm '+config.backward_log_path)
  if os.path.exists(config.use_output_path):
    os.system('rm '+config.use_output_path)
  for item in config.record_time:
    if os.path.exists(config.use_output_path+str(item)):
      os.system('rm '+config.use_output_path+str(item))
  if os.path.exists(config.use_log_path):
    os.system('rm '+config.use_log_path)
  if config.mode=='forward' or config.mode=='use':
    with tf.name_scope("forward_train"):
      with tf.variable_scope("forward", reuse=None):
        m_forward = PTBModel(is_training=True)
    with tf.name_scope("forward_test"):
      with tf.variable_scope("forward", reuse=True):
        mtest_forward = PTBModel(is_training=False)
    var=tf.trainable_variables()
    var_forward=[x for x in var if x.name.startswith('forward')]
    saver_forward=tf.train.Saver(var_forward, max_to_keep=1)
  if config.mode=='backward' or config.mode=='use':
    with tf.name_scope("backward_train"):
      with tf.variable_scope("backward", reuse=None):
        m_backward = PTBModel(is_training=True)
    with tf.name_scope("backward_test"):
      with tf.variable_scope("backward", reuse=True):
        mtest_backward = PTBModel(is_training=False)
    var=tf.trainable_variables()
    var_backward=[x for x in var if x.name.startswith('backward')]
    saver_backward=tf.train.Saver(var_backward, max_to_keep=1)
    
  init = tf.global_variables_initializer()
  

  with tf.Session() as session:
    session.run(init)
    if config.mode=='forward':
    	#train forward language model
      train_data, test_data = reader.read_data(config.data_path, config.num_steps)
      test_mean_old=15.0
      
      for epoch in range(config.max_epoch):
        train_ppl_list=[]
        test_ppl_list=[]
        for i in range(train_data.length//config.batch_size):
          input, sequence_length, target=train_data(m_forward.batch_size, i)
          train_perplexity = run_epoch(session, m_forward,input, sequence_length, target, mode='train')
          train_ppl_list.append(train_perplexity)
          print("Epoch:%d, Iter: %d Train NLL: %.3f" % (epoch, i + 1, train_perplexity))
        for i in range(test_data.length//config.batch_size):
          input, sequence_length, target=test_data(mtest_forward.batch_size, i)
          test_perplexity = run_epoch(session, mtest_forward, input, sequence_length, target, mode='test')
          test_ppl_list.append(test_perplexity)
          print("Epoch:%d, Iter: %d Test NLL: %.3f" % (epoch, i + 1, test_perplexity))
        test_mean=np.mean(test_ppl_list)
        if test_mean<test_mean_old:
          test_mean_old=test_mean
          saver_forward.save(session, config.forward_save_path)
        write_log('train ppl:'+str(np.mean(train_ppl_list))+'\t'+'test ppl:'+str(test_mean), config.forward_log_path)
    
    if config.mode=='backward':
    	#train backward language model
      train_data, test_data = reader.read_data(config.data_path, config.num_steps)
      test_mean_old=15.0
      for epoch in range(config.max_epoch):
        train_ppl_list=[]
        test_ppl_list=[]
      
        for i in range(train_data.length//config.batch_size):
          input, sequence_length, target=train_data(m_backward.batch_size, i)
          input, sequence_length, target=reverse_seq(input, sequence_length, target)
          train_perplexity = run_epoch(session, m_backward,input, sequence_length, target, mode='train')
          train_ppl_list.append(train_perplexity)
          print("Epoch:%d, Iter: %d Train NLL: %.3f" % (epoch, i + 1, train_perplexity))
        for i in range(test_data.length//config.batch_size):
          input, sequence_length, target=test_data(mtest_backward.batch_size, i)
          input, sequence_length, target=reverse_seq(input, sequence_length, target)
          test_perplexity = run_epoch(session, mtest_backward, input, sequence_length, target, mode='test')
          test_ppl_list.append(test_perplexity)
          print("Epoch:%d, Iter: %d Test NLL: %.3f" % (epoch, i + 1, test_perplexity))
        test_mean=np.mean(test_ppl_list)
        if test_mean<test_mean_old:
          test_mean_old=test_mean
          saver_backward.save(session, config.backward_save_path)
        write_log('train ppl:'+str(np.mean(train_ppl_list))+'\t'+'test ppl:'+str(test_mean), config.backward_log_path)
  
    if config.mode=='use':
      #CGMH sampling for sentence_correction
      sim=config.sim
      sta_vec=list(np.zeros([config.num_steps-1]))

      saver_forward.restore(session, config.forward_save_path)
      saver_backward.restore(session, config.backward_save_path)
      config.shuffle=False
      #erroneous sentence input
      if config.keyboard_input==True:
        #input from keyboard if key_input is not empty
        key_input=raw_input('please input a sentence\n')
        if key_input=='':
          use_data = reader.read_data_use(config.use_data_path, config.num_steps)
        else:
          sta_vec_list=[sen2sta_vec(key_input)]
          key_input=key_input.split()
          #key_input=sen2id(key_input)
          use_data = [key_input]
      else:
        #load keywords from file
        use_data=[]
        with open(config.use_data_path) as f:
          for line in f:
            use_data.append(line.strip().split())
      config.batch_size=1
      
      for sen_id in range(len(use_data)):
        #generate for each sentence
        input_=use_data[sen_id]
        pos=0

        for iter in range(config.sample_time):
        #ind is the index of the selected word, regardless of the beginning token.
          sta_vec=sen2sta_vec(' '.join(input_))
          input__=reader.array_data([sen2id(input_)], config.num_steps, config.dict_size)
          input,sequence_length,_=input__(1,0)
          input_original=input[0]

          ind=pos%(sequence_length[0]-1)
          print(' '.join(input_))
          
          if iter in config.record_time:
            with open(config.use_output_path+str(iter), 'a') as g:
              g.write(' '.join(input_)+'\n')
              
          if True: 
            prob_old=run_epoch(session, mtest_forward, input, sequence_length, mode='use')
            if config.double_LM==True:
              input_backward, _, _ =reverse_seq(input, sequence_length, input)
              prob_old=(prob_old+run_epoch(session, mtest_backward, input_backward, sequence_length, mode='use'))*0.5

            tem=1
            for j in range(sequence_length[0]-1):
              tem*=prob_old[0][j][input[0][j+1]]
            tem*=prob_old[0][j+1][config.dict_size+1]
            prob_old_prob=tem
            
            if sim!=None:
              similarity_old=similarity(input[0], input_original)
              prob_old_prob*=similarity_old
            else:
              similarity_old=-1
              
            input_candidate_=generate_change_candidate(input_, ind)
            tem=reader.array_data([sen2id(x) for x in input_candidate_], config.num_steps, config.dict_size)
            input_candidate,sequence_length_candidate,_=tem(len(input_candidate_),0)
            
            prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
            if config.double_LM==True:
              input_candidate_backward, _, _ =reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
              prob_candidate_pre=(prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5
            prob_candidate=[]
            for i in range(len(input_candidate_)):
              tem=1
              for j in range(sequence_length[0]-1):
                tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
              tem*=prob_candidate_pre[i][j+1][config.dict_size+1]
              prob_candidate.append(tem)
          
            prob_candidate=np.array(prob_candidate)
            if sim!=None:
              similarity_candidate=similarity_batch(input_candidate, input_original)
              prob_candidate=prob_candidate*similarity_candidate
            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
            prob_change_prob=prob_candidate[prob_candidate_ind]
            input_change_=input_candidate_[prob_candidate_ind]
          
          #word replacement (action: 0)
          if True: 
            if False:
              pass
            else:
              input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=0)
              prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
              prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
              prob_mul=(prob_forward*prob_backward)
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=1)
              prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
              if config.double_LM==True:
                input_candidate_backward, _, _ =reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
                prob_candidate_pre=(prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5

              prob_candidate=[]
              for i in range(config.search_size):
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                  tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                tem*=prob_candidate_pre[i][j+1][config.dict_size+1]
                prob_candidate.append(tem)
              prob_candidate=np.array(prob_candidate)
              if config.sim_word==True:
                similarity_candidate=similarity_batch(input_candidate[:,ind+1:ind+2], input_original[ind+1:ind+2])
                prob_candidate=prob_candidate*similarity_candidate
              prob_candidate_norm=normalize(prob_candidate)

              prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
              prob_candidate_prob=prob_candidate[prob_candidate_ind]
              
              prob_changeanother_prob=prob_candidate_prob
              word=id2sen(input_candidate[prob_candidate_ind])[ind]
              input_changeanother_=input_[:ind]+[word]+input_[ind+1:]
          
          #word insertion(action:1)
          if True: 
            if sequence_length[0]>=config.num_steps:
              prob_add_prob=0
              pass
            else:
              input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=1)
              prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
              prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
              prob_mul=(prob_forward*prob_backward)
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=1)
              prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
              if config.double_LM==True:
                input_candidate_backward, _, _ =reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
                prob_candidate_pre=(prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5

              prob_candidate=[]
              for i in range(config.search_size):
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                  tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                tem*=prob_candidate_pre[i][j+1][config.dict_size+1]
                prob_candidate.append(tem)
              prob_candidate=np.array(prob_candidate)
              #similarity_candidate=np.array([similarity(x, input_original) for x in input_candidate])
              if sim!=None:
                similarity_candidate=similarity_batch(input_candidate, input_original)
                prob_candidate=prob_candidate*similarity_candidate
              prob_candidate_norm=normalize(prob_candidate)

              prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
              prob_candidate_prob=prob_candidate[prob_candidate_ind]
              
              prob_add_prob=prob_candidate_prob
              word=id2sen(input_candidate[prob_candidate_ind])[ind]
              input_add_=input_[:ind]+[word]+input_[ind:]

        #word deletion(action: 2)
          if True:
            if sequence_length[0]<=2:
              prob_delete_prob=0
              pass
            else:
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, None , config.search_size, mode=2)
              prob_new=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
              tem=1
              for j in range(sequence_length_candidate[0]-1):
                tem*=prob_new[0][j][input_candidate[0][j+1]]
              tem*=prob_new[0][j+1][config.dict_size+1]
              prob_new_prob=tem
              if sim!=None:
                similarity_new=similarity_batch(input_candidate, input_original)
                prob_new_prob=prob_new_prob*similarity_new
              prob_delete_prob=prob_new_prob
            input_delete_=input_[:ind]+input_[ind+1:]
          b=np.argmax([prob_old_prob, prob_change_prob, prob_changeanother_prob*0.3, prob_add_prob*0.1, prob_delete_prob*0.001])
          print([prob_old_prob, prob_change_prob, prob_changeanother_prob, prob_add_prob, prob_delete_prob])
          print ([input_, input_change_,input_changeanother_, input_add_, input_delete_])
          input_=[input_, input_change_,input_changeanother_, input_add_, input_delete_][b]
          pos+=1
if __name__ == "__main__":
  tf.app.run()
