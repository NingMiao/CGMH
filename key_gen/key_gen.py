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
    #Build the inference graph using canonical LSTM cells.
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
  if os.path.exists(config.use_output_path):
    os.system('rm '+config.use_output_path)
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
  
  configs = tf.ConfigProto()
  configs.gpu_options.allow_growth = True
  with tf.Session(config=configs) as session:
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
      #CGMH sampling for key_gen
      sim=config.sim
      saver_forward.restore(session, config.forward_save_path)
      saver_backward.restore(session, config.backward_save_path)
      config.shuffle=False
      
      #keyword input
      if config.keyboard_input==True:
        #input from keyboard if key_input is not empty
        key_input=raw_input('please input a sentence\n')
        if key_input=='':
          use_data = reader.read_data_use(config.use_data_path, config.num_steps)
        else:
          key_input=key_input.split()
          key_input=sen2id(key_input)
          sta_vec=list(np.zeros([config.num_steps-1]))
          for i in range(len(key_input)):
            sta_vec[i]=1
          use_data = reader.array_data([key_input], config.num_steps, config.dict_size)
      else:
        #load keywords from file
        use_data, sta_vec_list = reader.read_data_use(config.use_data_path, config.num_steps)
      config.batch_size=1

      for sen_id in range(use_data.length):
        #generate for each sequence of keywords
        if config.keyboard_input==False:
          sta_vec=sta_vec_list[sen_id%(config.num_steps-1)]
        
        print(sta_vec)
       
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        
        pos=0
        outputs=[]
        output_p=[]
        for iter in range(config.sample_time):
          #ind is the index of the selected word, regardless of the beginning token.
          #sample config.sample_time times for each set of keywords
          config.sample_prior=[1,10.0/sequence_length[0],1,1]
          if iter%20<10:
            config.threshold=0
          else:
            config.threshold=0.5
          ind=pos%(sequence_length[0])
          action=choose_action(config.action_prob)
          print(' '.join(id2sen(input[0])))

          if sta_vec[ind]==1 and action in [0, 2]:                  
            #skip words that we do not change(original keywords)
            action=3
          
          #word replacement (action: 0)
          if action==0 and ind<sequence_length[0]-1: 
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
              similarity_old=similarity(input[0], input_original, sta_vec)
              prob_old_prob*=similarity_old
            else:
              similarity_old=-1
            input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=action)
            prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
            prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
            prob_mul=(prob_forward*prob_backward)
            input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=action)
            prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
            if config.double_LM==True:
              input_candidate_backward, _, _ =reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
              prob_candidate_pre=(prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5
            prob_candidate=[]
            for i in range(config.search_size):
              tem=1
              for j in range(sequence_length[0]-1):
                tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
              tem*=prob_candidate_pre[i][j+1][config.dict_size+1]
              prob_candidate.append(tem)
          
            prob_candidate=np.array(prob_candidate)
            if sim!=None:
              similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
              prob_candidate=prob_candidate*similarity_candidate
            prob_candidate_norm=normalize(prob_candidate)
            prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
            prob_candidate_prob=prob_candidate[prob_candidate_ind]
            if input_candidate[prob_candidate_ind][ind+1]<config.dict_size and ( prob_candidate_prob>prob_old_prob*config.threshold or just_acc()==0):
              input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
            pos+=1
            print('action:0', 1, prob_old_prob, prob_candidate_prob, prob_candidate_norm[prob_candidate_ind], similarity_old)
            if ' '.join(id2sen(input[0])) not in output_p:
              outputs.append([' '.join(id2sen(input[0])), prob_old_prob])
          
          #word insertion(action:1)
          if action==1: 
            if sequence_length[0]>=config.num_steps:
              action=3
            else:
              input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=action)
              prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
              prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
              prob_mul=(prob_forward*prob_backward)
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=action)
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
              prob_candidate=np.array(prob_candidate)*config.sample_prior[1]
              if sim!=None:
                similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
                prob_candidate=prob_candidate*similarity_candidate
              prob_candidate_norm=normalize(prob_candidate)

              prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
              prob_candidate_prob=prob_candidate[prob_candidate_ind]

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
                similarity_old=similarity(input[0], input_original,sta_vec)
                prob_old_prob=prob_old_prob*similarity_old
              else:
                similarity_old=-1
              #alpha is acceptance ratio of current proposal
              alpha=min(1, prob_candidate_prob*config.action_prob[2]/(prob_old_prob*config.action_prob[1]*prob_candidate_norm[prob_candidate_ind]))
              print ('action:1',alpha, prob_old_prob,prob_candidate_prob, prob_candidate_norm[prob_candidate_ind], similarity_old)
              if ' '.join(id2sen(input[0])) not in output_p:
                outputs.append([' '.join(id2sen(input[0])), prob_old_prob])
              if choose_action([alpha, 1-alpha])==0 and input_candidate[prob_candidate_ind][ind+1]<config.dict_size and (prob_candidate_prob>prob_old_prob* config.threshold or just_acc()==0):
                input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                sequence_length+=1
                pos+=2
                sta_vec.insert(ind, 0.0)
                del(sta_vec[-1])
              else:
                action=3
       
       
        #word deletion(action: 2)
          if action==2  and ind<sequence_length[0]-1:
            if sequence_length[0]<=2:
              action=3
            else:

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
                similarity_old=similarity(input[0], input_original,sta_vec)
                prob_old_prob=prob_old_prob*similarity_old
              else:
                similarity_old=-1
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, None , config.search_size, mode=2)
              prob_new=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
              tem=1
              for j in range(sequence_length_candidate[0]-1):
                tem*=prob_new[0][j][input_candidate[0][j+1]]
              tem*=prob_new[0][j+1][config.dict_size+1]
              prob_new_prob=tem
              if sim!=None:
                similarity_new=similarity_batch(input_candidate, input_original,sta_vec)
                prob_new_prob=prob_new_prob*similarity_new
            
              input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=0)
              prob_forward=run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%(sequence_length[0]-1),:]
              prob_backward=run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0]-1-ind%(sequence_length[0]-1),:]
              prob_mul=(prob_forward*prob_backward)
              input_candidate, sequence_length_candidate=generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=0)
              prob_candidate_pre=run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')
              if config.double_LM==True:
                input_candidate_backward, _, _ =reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
                prob_candidate_pre=(prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5

              prob_candidate=[]
              for i in range(config.search_size):
                tem=1
                for j in range(sequence_length[0]-1):
                  tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                tem*=prob_candidate_pre[i][j+1][config.dict_size+1]
                prob_candidate.append(tem)
              prob_candidate=np.array(prob_candidate)
            
              if sim!=None:
                similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec)
                prob_candidate=prob_candidate*similarity_candidate
              
              #alpha is acceptance ratio of current proposal
              prob_candidate_norm=normalize(prob_candidate)
              if input[0] in input_candidate:
                for candidate_ind in range(len(input_candidate)):
                  if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                    break
                  pass
                alpha=min(prob_candidate_norm[candidate_ind]*prob_new_prob*config.action_prob[1]/(config.action_prob[2]*prob_old_prob), 1)
              else:
                pass
                alpha=0
              print('action:2', alpha, prob_old_prob, prob_new_prob, prob_candidate_norm[candidate_ind], similarity_old)
              if ' '.join(id2sen(input[0])) not in output_p:
                outputs.append([' '.join(id2sen(input[0])), prob_old_prob])
              if choose_action([alpha, 1-alpha])==0 and (prob_new_prob> prob_old_prob*config.threshold or just_acc()==0):
                input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+config.dict_size+1], axis=1)
                sequence_length-=1
                pos+=0
                del(sta_vec[ind])
                sta_vec.append(0)
              else:
                action=3
          #skip word (action: 3)
          if action==3:
            #write_log('step:'+str(iter)+'action:3', config.use_log_path)
            pos+=1
          print(outputs)
          if outputs !=[]:
            output_p.append(outputs[-1][0])
        
        #choose output from samples
        for num in range(config.min_length, 0, -1):
          outputss=[x for x in outputs if len(x[0].split())>=num]
          print(num,outputss)
          if outputss!=[]:
            break
        if outputss==[]:
          outputss.append([' '.join(id2sen(input[0])),1])
        outputss=sorted(outputss, key=lambda x: x[1])[::-1]
        with open(config.use_output_path, 'a') as g:
          g.write(outputss[0][0]+'\n')

if __name__ == "__main__":
  tf.app.run()
