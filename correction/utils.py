from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from config import config
config=config()

import pickle as pkl
if config.sim=='word_max' or config.sim=='combine' or config.sim_word==True:
    emb_word,emb_id=pkl.load(open(config.emb_path))

import sys
sys.path.insert(0,config.skipthoughts_path)
sys.path.insert(0,'../utils/dict_emb')
from dict_use import dict_use
dict_use=dict_use(config.dict_path)
sen2id=dict_use.sen2id
id2sen=dict_use.id2sen
dic,dic_reverse=pkl.load(open(config.dict_path))
sys.path.insert(0,config.liguistic_path)
import en
if config.sim=='skipthoughts' or config.sim=='combine':
    import skipthoughts
    skip_model = skipthoughts.load_model()
    skip_encoder = skipthoughts.Encoder(skip_model)


if config.keyword_pos==True:
    import RAKE
    Rake = RAKE.Rake(RAKE.SmartStopList())
    from zpar import ZPar
    z=ZPar(config.pos_path)
    tagger = z.get_tagger()
    

def normalize(x, e=0.05):
    tem = x
    if max(tem)==0:
        tem+=e
    return tem/tem.sum()

def reverse_seq(input, sequence_length, target):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_new=np.zeros([batch_size, num_steps])+config.dict_size+1
    target_new=np.zeros([batch_size, num_steps])+config.dict_size+1
    for i in range(batch_size):
        length=sequence_length[i]-1
        for j in range(length):
            target_new[i][j]=target[i][length-1-j]
        input_new[i][0]=config.dict_size+2
        for j in range(length):
            input_new[i][j+1]=input[i][length-j]
    return input_new.astype(np.int32), sequence_length.astype(np.int32), target_new.astype(np.int32)
#
def cut_from_point(input, sequence_length, ind, mode=0):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_forward=np.zeros([batch_size, num_steps])+config.dict_size+1
    input_backward=np.zeros([batch_size, num_steps])+config.dict_size+1
    sequence_length_forward=np.zeros([batch_size])
    sequence_length_backward=np.zeros([batch_size])
    for i in range(batch_size):
        input_forward[i][0]=config.dict_size+2
        input_backward[i][0]=config.dict_size+2
        length=sequence_length[i]-1

        for j in range(ind):
            input_forward[i][j+1]=input[i][j+1]
        sequence_length_forward[i]=ind+1
        if mode==0:
            for j in range(length-ind-1):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind
        elif mode==1:
            for j in range(length-ind):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind+1
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(np.int32), sequence_length_backward.astype(np.int32)
    
def generate_candidate_input(input, sequence_length, ind, prob, search_size, mode=0):
    input_new=np.array([input[0]]*search_size)
    sequence_length_new=np.array([sequence_length[0]]*search_size)
    length=sequence_length[0]-1
    if mode!=2:
        ind_token=np.argsort(prob[: config.dict_size])[-search_size:]
    
    if mode==2:
        for i in range(sequence_length[0]-ind-2):
            input_new[: , ind+i+1]=input_new[: , ind+i+2]
        for i in range(sequence_length[0]-1, config.num_steps-1):
            input_new[: , i]=input_new[: , i]*0+config.dict_size+1
        sequence_length_new=sequence_length_new-1
        return input_new[:1], sequence_length_new[:1]
    if mode==1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[: , sequence_length_new[0]-i]=input_new[: ,  sequence_length_new[0]-1-i]
        sequence_length_new=sequence_length_new+1
    for i in range(search_size):
        input_new[i][ind+1]=ind_token[i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)

def sample_from_candidate(prob_candidate):
    return np.argmax(prob_candidate)
''' 
def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))
'''
def choose_action(c):
    r=np.random.random()
    c=np.array(c)
    for i in range(1, len(c)):
        c[i]=c[i]+c[i-1]
    for i in range(len(c)):
        if c[i]>=r:
            return i

def sentence_embedding(s):
    emb_sum=0
    cou=0
    for item in s:
        if item<config.dict_size:
            emb_sum+=emb[item]
            cou+=1
    return emb_sum/(cou+0.0001)
'''
def similarity(s1, s2):
    e1=sentence_embedding(s1)
    e2=sentence_embedding(s2)
    cos=(e1*e2).sum()/((e1**2).sum()*(e2**2).sum())**0.5
    return cos**config.sim_hardness
'''
if config.sim=='skipthoughts' or config.sim=='combine':
    def sigma_skipthoughts(x):
        return (np.abs(1-((x-1)*2)**2)+(1-((x-1)*2)**2))/2.0
        #return 1
    def similarity_skipthoughts(s1, s2):
        #s2 is reference_sentence
        s1=' '.join(id2sen(s1))
        s2=' '.join(id2sen(s2))
        #print(s1,s2)
        e=skip_encoder.encode([s1,s2])
        e1=e[0]
        e2=e[-1]
        cos=(e1*e2).sum()/((e1**2).sum()*(e2**2).sum())**0.5
        return sigma_skipthoughts(cos)
    def similarity_batch_skipthoughts(s1, s2):
        #s2 is reference_sentence
        s1=[' '.join(id2sen(x)) for x in s1]
        s2=' '.join(id2sen(s2))
        s1.append(s2)
        e=skip_encoder.encode(s1)
        e1=e[:-1]
        e2=e[-1]
        cos=(e1*e2).sum(axis=1)/((e1**2).sum(axis=1)*(e2**2).sum())**0.5
        return sigma_skipthoughts(cos)

if config.sim=='word_max' or config.sim=='combine' or config.sim_word==True:
    def sigma_word(x):
        return 1
        '''
        if x>0.6:
            return x
        elif x>0.5:
            return 6*(x-0.5)
        else:
            return 0
        '''
    def sen2mat(s):
        mat=[]
        for item in s:
            if item==config.dict_size+2:
                continue
            if item==config.dict_size+1:
                break
            if item in emb_id:
                mat.append(np.array(emb_id[item]))
            else:
                mat.append(np.random.random([config.hidden_size]))
        return np.array(mat)
    def similarity_word(s1,s2):
        e=1e-5
        emb1=sen2mat(s1)
        #wei2=normalize( np.array([-np.log(id2freq[x]) for x in s2 if x<=config.dict_size]))
        emb2=sen2mat(s2)
        emb_mat=np.dot(emb2,emb1.T)
        norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e))
        norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e))
        sim_mat=np.dot(norm2,emb_mat).dot(norm1)
        sim_vec=sim_mat.max(axis=1)
        #sim=(sim_vec*wei2).sum()
        sim=(sim_vec).mean()
        return sigma_word(sim)
    
    def similarity_batch_word(s1, s2):
        return np.array([ similarity_word(x,s2) for x in s1 ])

if config.sim=='skipthoughts':
    similarity=similarity_skipthoughts
    similarity_batch=similarity_batch_skipthoughts
elif config.sim=='word_max' or config.sim_word==True:
    similarity=similarity_word
    similarity_batch=similarity_batch_word
elif config.sim=='combine':
    def similarity(s1,s2):
        return (similarity_skipthoughts(s1, s2)+similarity_word(s1, s2))/2.0
    def similarity_batch(s1,s2):
        return (similarity_batch_skipthoughts(s1, s2)+similarity_batch_word(s1, s2))/2.0

def keyword_pos2sta_vec(keyword, pos):
    key_ind=[]
    pos=pos[:config.num_steps-1]
    for i in range(len(pos)):
        if pos[i]=='NNP':
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS'] and keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['VBZ'] and keyword[i]==1:
            key_ind.append(i)
        elif keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS','VBZ']:
            key_ind.append(i)
    key_ind=key_ind[:max(int(config.max_key_rate*len(pos)), config.max_key)]
    sta_vec=[]
    for i in range(len(keyword)):
        if i in key_ind:
            sta_vec.append(1)
        else:
            sta_vec.append(0)
    return sta_vec
    
def sen2sta_vec(line):
    sta_vec=list(np.zeros([config.num_steps-1]))
    if config.keyword_pos==False:
        return sta_vec
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
    keyword_pos2sta_vec(sta_vec,pos)
  
def generate_change_candidate(input_, ind):
    input_candidate_=[]
    input_candidate_.append(input_)
    word=input_[ind]
    suggest_word=[]
    if word not in dic:
        suggest_word.extend(en.spelling.suggest(word))
    try:
        suggest_word.append(en.noun.plural(word))
        suggest_word.append(en.noun.singular(word))
    except:
        pass
    try:
        suggest_word.append(en.verb.infinitive(word))
        suggest_word.append(en.verb.present(word))
        suggest_word.append(en.verb.present(word, person=3, negate=False))
        suggest_word.append(en.verb.present_participle(word))
        suggest_word.append(en.verb.past(word, person=3, negate=False))
        suggest_word.append(en.verb.past_particip(word))
    except:
        pass
    suggest_word_new=[x for x in suggest_word if x in dic][:config.max_suggest_word-1]
    
    for item in suggest_word_new:
        input_candidate_.append(input_[:ind]+[item]+input_[ind+1:])
    return input_candidate_
    
def just_acc():
    r=np.random.random()
    if r<config.just_acc_rate:
        return 0
    else:
        return 1
        
def write_log(str, path):
  with open(path, 'a') as g:
    g.write(str+'\n')