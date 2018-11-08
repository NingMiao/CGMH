class config(object):
    def __init__(self):
        self.data_path='../data/1-billion/1-billion.txt'                         #path of data for training language model
        self.use_data_path='./input/sen.txt'                                        #data path of erroneous sentence for correction
        self.dict_path='../data/1-billion/dict.pkl'                                 #dictionary path
        self.emb_path='../data/1-billion/emb.pkl'                               #word embedding path, used when config.sim=='word_max' or config.sim=='combine'
        self.skipthoughts_path='../skip_thought'                                 #path of skipthoughts, used when config.sim=='skipthoughts' or config.sim=='combine'        
        self.liguistic_path='../linguistics'                                              #path of data of liguistics package
        self.pos_path='../POS/english-models'                                    #path for pos tagger
        self.dict_size=50000
        self.vocab_size=50003
        self.forward_save_path='./model/forward.ckpt'
        self.backward_save_path='./model/backward.ckpt'
        self.forward_log_path='./log/forward_log.txt'
        self.backward_log_path='./log/backward_log.txt'
        self.use_fp16=False
        self.shuffle=False
        self.use_log_path='./log/use_log.txt'
        
        
        self.batch_size=32
        self.num_steps=50
        self.hidden_size=300
        
        self.keep_prob=1
        self.num_layers=2
        
        self.max_epoch=100
        self.max_grad_norm=5
        
        self.GPU='0'
        self.mode='use'
        self.sample_time=31
        self.record_time=[30]
        for i in range(len(self.record_time)):
            self.record_time[i]*=2                                                      #output the sentence at record_time steps
        self.sample_sentence_number=119
        
        self.search_size=100
        self.use_output_path='./output/output.txt'                        #output path
      
        
        self.action_prob=[0.3,0.3,0.3,0.1]                                          #the prior of 4 actions
        self.threshold=0.1
        self.sim=None                                                                      #matching model
        self.sim_word=True
        self.double_LM=False
        self.keyword_pos=False
        self.keyboard_input=False                                                     #input from keyboard
        self.sim_hardness=5
        self.rare_since=30000
        self.just_acc_rate=0.0
        self.max_key=3
        self.max_key_rate=0.5
        self.max_suggest_word=20