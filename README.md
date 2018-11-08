# Constrained Sentence Generation via Metropolis-Hastings Sampling
## Introduction ##
CGMH is a sampling based model for constrained sentence generation, which can be used in keyword-to-sentence generation, paraphrase, sentence correction and many other tasks.
## Examples ##
- Running example for parahrase:  (All rejected proposal is omitted)  
  what movie do you like most . ->  
  which movie do you like most . (`replace` what `with` which) ->  
  which movie do you like . (`delete` most) ->  
  which movie do you like best . (`insert` best) ->  
  which movie do you think best . (`replace` like `with` think) ->  
  which movie do you think the best . (`insert` the) ->  
  which movie do you think is the best . (`insert` is)  
  
- Running example for sentence correction:
  in the word oil price very high right now . ->  
  in the word , oil price very high right now . (`insert` ,) ->  
  in the word , oil prices very high right now . (`replace` price `with` prices) ->  
  in the word , oil prices are very high right now . (`insert` are)

- Extra Examples for sentence correction:  
  origin: even if we are failed , we have to try to get a new things .->  
  generated: even if we are failing , we have to try to get some new things .  

  origin: in the word oil price very high right now .->  
  generated: in the word , oil prices are very high right now .  

  origin: the reason these problem occurs is also becayse of the exam .->  
  generated: the reason these problems occur is also because of the exam .


## Requirement ##
- python
  - `==2.7`

- TensorFlow
  - `== 1.3.0`
  
- python packages
  - numpy
  - pickle
  - Rake
  - zPar
  - skipthoughts
  
- word embedding
  - If you want to try using word embedding for paraphrase, you should download or train a word embedding first and place it at config.emb_path.

## Running ##
- Training language models
  - For each task, first train a backward and a language model:  
      set `mode='forward'` and `mode='backward'` in `config.py` successively.  
      run `crrection.py` / `paraphrase.py` / `key-gen.py` to train each model  
    
- Generation
  - For generating new sample for each tasks:  
      set `mode='use'` and choose proper parameter in `config.py`.   
      run `crrection.py` / `paraphrase.py` / `key-gen.py` to sample.  
      outputs are in `output`.
