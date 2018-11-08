import pickle as pkl
class dict_use:
    def __init__(self, dict_path):
        self.Dict1, self.Dict2=pkl.load(open(dict_path,'rb'))
        self.vocab_size=len(self.Dict1)+3
        self.UNK=self.vocab_size-3
        self.BOS=self.vocab_size-1
        self.EOS=self.vocab_size-2
        
    def sen2id(self, s):
        if s==[]:
            return []
        Dict=self.Dict1
        dict_size=len(Dict)
        s_new=[]
        if type(s[0])!=type([]):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                else:
                    s_new.append(dict_size)
            return s_new
        else:
            return [self.sen2id(x) for x in s]
    
    def id2sen(self, s):
        if s==[]:
            return []
        Dict=self.Dict2
        dict_size=len(Dict)
        s_new=[]
        if type(s[0])!=type([]):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                elif item==dict_size:
                    s_new.append( 'UNK')
                else:
                    pass
            return s_new
        else:
            return [self.id2sen(x) for x in s]