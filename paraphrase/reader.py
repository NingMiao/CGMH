from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle as pkl
from config import config
config=config()
from utils import *
import sys
sys.path.insert(0,config.dict_path)
from dict_use import *
import RAKE
Rake = RAKE.Rake(RAKE.SmartStopList())
from zpar import ZPar
z=ZPar(config.pos_path)
tagger = z.get_tagger()
tt_proportion=0.9

class dataset(object):
    def __init__(self, input, sequence_length, target ):
        self.input=input
        self.target=target
        self.sequence_length=sequence_length
        self.length=len(input)
    def __call__(self, batch_size, step):
        batch_num=self.length//batch_size
        step=step%batch_num
        return self.input[step*batch_size: (step+1)*batch_size], self.sequence_length[step*batch_size: (step+1)*batch_size], self.target[step*batch_size: (step+1)*batch_size]
        
def array_data(data,  max_length, dict_size, shuffle=False):
    max_length_m1=max_length-1
    if shuffle==True:
        np.random.shuffle(data)
    sequence_length_pre=np.array([len(line) for line in data]).astype(np.int32)
    sequence_length=[]
    for item in sequence_length_pre:
        if item>max_length_m1:
            sequence_length.append(max_length)
        else:
            sequence_length.append(item+1)
    sequence_length=np.array(sequence_length)
    for i in range(len(data)):
        if len(data[i])>=max_length_m1:
            data[i]=data[i][:max_length_m1]
        else:
            for j in range(max_length_m1-len(data[i])):
                data[i].append(dict_size+1)
        data[i].append(dict_size+1)
    target=np.array(data).astype(np.int32)
    input=np.concatenate([np.ones([len(data), 1])*(dict_size+2), target[:, :-1]], axis=1).astype(np.int32)
    return dataset(input, sequence_length, target)
        
def read_data(file_name,  max_length, dict_size=config.dict_size):
	if file_name[-3:]=='pkl':
	    data=pkl.load(open(file_name))
	else:
	    with open(file_name) as f:
	        data=[]
	        for line in f:
	            data.append(sen2id(line.strip().split()))
	train_data=array_data(data[ : int(len(data)*tt_proportion)], max_length, dict_size, shuffle=True)
	test_data=array_data(data[int(len(data)*tt_proportion) : ], max_length, dict_size, shuffle=True)
	return train_data, test_data
	
def read_data_use(file_name,  max_length,dict_size=config.dict_size):
    if file_name[-3:]=='pkl':
        data=pkl.load(open(file_name))
        sta_vec_list=[list(np.zeros([config.num_steps-1]))]*len(data)                           ############some remaining problems here
    else:
        with open(file_name) as f:
            data=[]
            vector=[]
            sta_vec_list=[]
            j=0
            for line in f:
                sta_vec=list(np.zeros([config.num_steps-1]))
                keyword=Rake.run(line.strip())
                pos_list=tagger.tag_sentence(line.strip()).split()
                pos=zip(*[x.split('/') for x in pos_list])[0]
                if keyword!=[]:
                    keyword=list(zip(*keyword)[0])
                    keyword_new=[]
                    for item in keyword:
                        tem1=[line.strip().split().index(x) for x in item.split() if x in line.strip().split()]
                        keyword_new.extend(tem1)
                    for i in range(len(keyword_new)):
                        ind=keyword_new[i]
                        if ind<config.num_steps-2:
                            sta_vec[ind]=1
                if config.keyword_pos==True:
                    sta_vec_list.append(keyword_pos2sta_vec(sta_vec,pos))
                else:
                    sta_vec_list.append(list(np.zeros([config.num_steps-1])))
                data.append(sen2id(line.strip().lower().split()))
    data_new=array_data(data, max_length, dict_size)
    return data_new, sta_vec_list