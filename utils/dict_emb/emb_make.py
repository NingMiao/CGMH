import numpy as np
import pickle as pkl

glove_path='../../data/glove/glove.840B.300d.txt'
dict_path='../../data/1-billion/dict.pkl'
emb_path='../../data/1-billion/emb.pkl'

f=open(glove_path)
emb=[]
i=0

for line in f:
    line=line.strip().split()
    word=line[0]
    #word_emb=np.array([float(x) for x in line[1:]])
    word_emb=np.array([float(x) for x in line[1:]])
    emb.append([word, word_emb])
    if i%100==0:
        print i
    i+=1

emb=dict(emb)
print 'emb loading finished'

word_dict , id_dict =pkl.load(open(dict_path))
word_list=word_dict.keys()
emb_small=[]
emb_small_id=[]
i=0
for i in range(len(word_list)):
    word=id_dict[i]
    if word in emb:
        emb_small.append([word, emb[word]])
        emb_small_id.append([ i, emb[word]])
    else:
        emb_small.append([word, np.random.random([300])])
        emb_small_id.append([i, np.random.random([300])])
    if i%100==0:
        print i
    i+=1

emb_small.append(['UNK',np.random.random([300])])
emb_small_id.append([i,np.random.random([300])])
with open(emb_path ,'w') as g:
    pkl.dump([dict(emb_small), dict(emb_small_id)], g)