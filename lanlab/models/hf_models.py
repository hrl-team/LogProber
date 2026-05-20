import torch
from flask import Flask, request
import logging
#import torch.multiprocessing as mp
from pathos.helpers import mp
import numpy as np
import os
import time

from lanlab.models.model import Model
from lanlab.models.openai_models import sequence_from_OPENAICompletion
from lanlab.data_management.batch.sequence import Sequence
from lanlab.models.model import ModelConfig
import requests

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

#Test
"""try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
def dict_to_openai(token_ids,logits,tokenizer,temperature,stop,return_logits=False,nb_logprobs=5,inputs_to_remove=None):
    """ Returns the data in the OPENAI format"""
    #Compute logp associated with these ids
    logprobs = (logits/temperature).softmax(-1).log()
    generated_tokens_logp = [lprobs[token_id].item() for lprobs,token_id in zip(logprobs,token_ids[1:])]

    #Translate the token ids into text
    generated_tokens = [tokenizer.convert_ids_to_tokens([token_id])[0].replace('▁',' ').replace('Ġ',' ') for token_id in token_ids]
    generated_sequence = (''.join(generated_tokens))
    
    #Top logp computations
    top_logp = []
    for lprobs in logprobs:
        best_token_ids = lprobs.argsort()[-nb_logprobs:]
        top_logp.append({tokenizer.convert_ids_to_tokens([token_id.item()])[0].replace('▁',' ').replace('Ġ',' '):lprobs[token_id.item()].item() for token_id in best_token_ids})

    #Outputs
    tokens_logprobs = [None] + generated_tokens_logp
    top_logprobs = [None] + top_logp

    #Compute echo
    if not(inputs_to_remove is None):
        generated_tokens = generated_tokens[inputs_to_remove.shape[0]:]
        generated_sequence = ''.join(generated_tokens)

        tokens_logprobs = tokens_logprobs[inputs_to_remove.shape[0]:]
        top_logprobs = top_logprobs[inputs_to_remove.shape[0]:]
        
        
    #Return the dict
    out =  {'choices':[
        {'text':generated_sequence,
         'logprobs':{
             'tokens': generated_tokens,
             'token_logprobs': tokens_logprobs,
             'top_logprobs': top_logprobs,
         },
         'finish_reason':stop
        }
    ]
    }
    if return_logits:
        out['choices'][0]['logprobs']['logits'] = logits.numpy().astype(np.float16).tolist()
    return out


def default_config():
    """ Default config for the model generation"""
    return {'prompt':None,
            'temperature':0.7,
            'min_tokens':0,
            'max_tokens':8,
            'logprobs':5,
            'stop':[],
            'echo':False,
            'return_logits':False}

def flask_server(port,inp_queue):
    """ Starts a flask server that will handle the requests"""
    app = Flask(__name__)

    @app.route("/completions", methods=['POST'])
    def completions():
        """ Flask route for the completions"""
        global server
        logging.debug('got request')
        config = default_config()
        for k in config:
            try:
                config[k] = request.json[k]
            except KeyError:
                pass
        logging.debug('parsed request')
        parent_conn,child_conn = mp.Pipe()
        inp_queue.put({"config":config,"pipe":child_conn})
        logging.debug('waiting for the model')
        out = parent_conn.recv()#server.ret_queue.get()
        logging.debug('returns')
        return out
    
    app.run(host='0.0.0.0',port=port,threaded=True)
    
class Server:
    """ Server that handles the requests"""
    def __init__(self,model_loader,port):
        self.port = port
        self.inp_queue = mp.Queue()
        self.ret_queue = mp.Queue()
        self.model_loader = model_loader
        self.active = False

        self.batch_size = 64
        self.timeout = 1

        self.batch = []
        self.last_time = 0

    def start(self,wait=False):
        """ Starts the server and creates the process that will handle the requests"""
        
        logging.info('starting flask server')
        self.flask = mp.Process(target=flask_server,args=(self.port,self.inp_queue))
        self.flask.start()
        
        logging.info('starting model process')
        self.process = mp.Process(target=completion_loop,args=(self.model_loader,self.inp_queue,self.ret_queue,self.timeout,self.batch_size))
        self.process.start()
        self.active = True
        if wait:
            logging.info('waiting for the model to load on the remote server')
            self.wait_for_start()

    def wait_for_start(self):
        self.ret_queue.get() #== 'loaded'

    def stop(self):
        self.flask.terminate()
        self.process.terminate()
        self.active = False

    def load_model(self):
        raise NotImplementedError

def completion_loop(model_loader,inp_queue,out_queue,timeout,batch_size):
    """ Loop that handles the requests and sends them to the model"""
    logging.info('loading model')
    tokenizer,model = model_loader()
    logging.info('model loaded')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    last_time = time.perf_counter()
    batch = []

    def flush(batch):
        """Flushes the batch and sends it to the model and resets the timer"""
        if len(batch)>=1:
            logging.debug('flush batch of size '+str(len(batch)))
            complete(tokenizer,model,batch)
            logging.debug('completed batch')

    out_queue.put('loaded')
    logging.debug('model loaded')

    while True:
        #Get the first item in the queue
        if not(inp_queue.empty()):
            top = inp_queue.get(timeout=1)
            if isinstance(top,dict):
                logging.debug('got completion order')
                batch.append(top)
                if len(batch)>=batch_size:
                    flush(batch)
                    last_time = time.perf_counter()
                    batch = []
        else: #Pretty bad but doesn't work if I try catch empty directly
            if time.perf_counter()-last_time>timeout:
                flush(batch)
                last_time = time.perf_counter()
                batch = []
        time.sleep(0.1)


def generate(model,tokenizer,inp_batch,min_tokens,max_tokens,temperature=0.7,stop=None,pad_token_id=3):
    """Generates text from the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Process the full input first
    token_ids = inp_batch.to(device)
    generated_text = ['' for i in range(len(token_ids))]
    out = model(token_ids)
    logits,kv = out[0].cpu(),out[1]
    if stop is None:
        stop = [[]]*token_ids.shape[0]
    stopped = [False]*len(token_ids)
    count = 0
    while True: #Continue the text
        count += 1
        #Compute probs for the next token
        new_token_ids = []
        for i in range(len(temperature)):
            if temperature[i] > 0:
                probs = (logits[i][-1]/temperature[i]).softmax(-1).double()
                
                #Sample the next token from probs
                dist = torch.distributions.Categorical(probs)
                new_token_id = dist.sample()[None,None]
            else:
                probs = logits
                new_token_id = logits[i][-1].argmax()[None,None]
                #token_ids = torch.cat([token_ids,new_token_id],dim=1)

            if stopped[i]:
                new_token_id = torch.tensor([[pad_token_id]],dtype=torch.int64) #pad if it has already been stopped -> in bloom pad token is 3

            generated_text[i] = tokenizer.convert_ids_to_tokens(new_token_id)

            #Verify stop conditions
            # -- Max tokens
            if len(token_ids[i]) >= max_tokens[i]+len(inp_batch[i]):
                stopped[i] = 'length'
            # -- sequence stop condition
            f = False
            for s in stop[i]: #Check all the stop conditions
                if s in generated_text[i]: #If one is found, stop
                    f = True
            if f:
                stopped[i] = 'stop_condition'

            #Add the new token to the sequence
            new_token_ids.append(new_token_id)

        #Stop the loop if all generations are stopped
        if np.all(np.array(stopped,bool)):
            break

        #Compute logits from this new token
        new_token_ids = torch.cat(new_token_ids).to(device)
        out = model(new_token_ids,past_key_values=kv)
        new_logits,kv = out[0].cpu(),out[1]
        logits = [torch.cat([logit,new_logit],dim=0) if not(s) else logit for logit,new_logit,s in zip(logits,new_logits,stopped)]
        token_ids = [torch.cat([token_id,new_token_id],dim=0) if not(s) else token_id for token_id,new_token_id,s in zip(token_ids,new_token_ids,stopped)]
    return token_ids,logits,stopped
    
def complete(tokenizer,model,data):
    """Completes the queries given in data and returns the results with the OPENAI format"""
    configs,pipes = [data_['config'] for data_ in data],[data_['pipe'] for data_ in data]
    prompts = [config['prompt'] for config in configs]
    inputs = tokenizer(prompts,return_tensors='pt',padding=True).input_ids #Put back padding at True to handle multiple query at once
    temperature = [config['temperature'] for config in configs]
    min_tokens = [config['min_tokens'] for config in configs]
    max_tokens = [config['max_tokens'] for config in configs]
    stop = [config['stop'] if not(config['stop'] is None) else [] for config in configs]
    echo = [config['echo'] for config in configs]
    nb_logprobs = [config['logprobs'] for config in configs]
    return_logits = [config['return_logits'] for config in configs]
    
    with torch.no_grad():
        token_ids,logits,stopped = generate(model,tokenizer,inputs,min_tokens,max_tokens,temperature=temperature,stop=stop)
    results = [dict_to_openai(token_ids[i],logits[i],tokenizer,temperature[i],stopped[i],return_logits=return_logits[i],nb_logprobs=nb_logprobs[i],inputs_to_remove=inputs[i] if not echo[i] else None) for i in range(len(configs))]
    for r,p in zip(results,pipes):
        p.send(r)
        p.close()

class HFModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.add_key('return_logits',False)

class HFModel(Model):
    def __init__(self,port=None,model_path=None):
        if port is None:
            port = np.random.randint(10000,50000)
        self.model_path = model_path
        super().__init__()
        self.port = port
        self.server = Server(self.load_model,port=self.port)
        

    def start_server(self):
        self.server.start(wait=True)

    def close(self):
        if self.server.active:
            self.server.stop()
        super().close()

    @property
    def config_class(self):
        return HFModelConfig
    
    def complete(self,sequence,config=None):
        if not(self.server.active):
            self.start_server()
        if config is None:
            config = self.config
        prompt = str(sequence)
        data = {'prompt':prompt,'logprobs':5,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        if self['return_logits']:
            answer['logits'] = np.array(answer['choices'][0]['logprobs']['logits'],np.float16)
        segment = sequence_from_OPENAICompletion(answer)
        return sequence + segment

    def read(self,sequence,config=None):
        if not(self.server.active):
            self.start_server()
        if config is None:
            config = self.config
        prompt = str(sequence)
        config['max_tokens'] = 0
        data = {'prompt':prompt,'logprobs':5,'echo':True,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        sequence = sequence_from_OPENAICompletion(answer)
        return sequence

class AutoModel(HFModel):
    def load_model(self):

        #Load the model
        name = self.name

        from transformers import AutoModel, AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        hf_tokenizer.pad_token = '<s>'

        #if 't5' in name:
        #    hf_model = T5ForConditionalGeneration.from_pretrained(os.path.join('..','..','model',name)).to(device)
        if 'stablelm' in name:
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(self.model_path)
        elif 'Cerebras' in name:
            from transformers import GPT2LMHeadModel
            hf_model = GPT2LMHeadModel.from_pretrained(self.model_path)
        elif 'pythia' in name or 'dolly' in name:
            from transformers import GPTNeoXForCausalLM
            hf_model = GPTNeoXForCausalLM.from_pretrained(self.model_path)
        elif 'bloom' in name or 'phoenix-inst-chat' in name:
            from transformers import BloomForCausalLM
            hf_model = BloomForCausalLM.from_pretrained(self.model_path)
        else:# 'llama' in name or 'checkpoint' in name or 'alpaca' in name or 'wizard' in name or 'vicuna' in name or 'chimera' in name or 'baize' in name or 'guanaco' in name:
            from transformers import LlamaForCausalLM
            hf_model = LlamaForCausalLM.from_pretrained(self.model_path)
        #else:
        #    from transformers import AutoModel
        #    hf_model = AutoModel.from_pretrained(self.model_path)

        return hf_tokenizer,hf_model

    @property
    def name(self):
        return self.model_path.split('/')[-1]

class AutoUnslothModel(HFModel):
    def load_model(self):

        #Load the model
        name = self.name

        from unsloth import FastLanguageModel
        from transformers import AutoModel, AutoTokenizer
        
        #hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        #hf_tokenizer.pad_token = '<s>'

        print(self.model_path)

        hf_model, hf_tokenizer = FastLanguageModel.from_pretrained(
            self.model_path,load_in_4bit=False,resize_model_vocab = 151666, local_files_only=True)
        hf_model = FastLanguageModel.for_inference(hf_model)

        print('--------')
        
        #if 't5' in name:
        #    hf_model = T5ForConditionalGeneration.from_pretrained(os.path.join('..','..','model',name)).to(device)
        '''if 'stablelm' in name:
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(self.model_path)
        elif 'Cerebras' in name:
            from transformers import GPT2LMHeadModel
            hf_model = GPT2LMHeadModel.from_pretrained(self.model_path)
        elif 'pythia' in name or 'dolly' in name:
            from transformers import GPTNeoXForCausalLM
            hf_model = GPTNeoXForCausalLM.from_pretrained(self.model_path)
        elif 'bloom' in name or 'phoenix-inst-chat' in name:
            from transformers import BloomForCausalLM
            hf_model = BloomForCausalLM.from_pretrained(self.model_path)
        else:# 'llama' in name or 'checkpoint' in name or 'alpaca' in name or 'wizard' in name or 'vicuna' in name or 'chimera' in name or 'baize' in name or 'guanaco' in name:
            from transformers import LlamaForCausalLM
            hf_model = LlamaForCausalLM.from_pretrained(self.model_path)'''
        #else:
        #    from transformers import AutoModel
        #    hf_model = AutoModel.from_pretrained(self.model_path)

        return hf_tokenizer,hf_model

    @property
    def name(self):
        return self.model_path.split('/')[-1]

class Llama7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'llama-7b'
    
class Llama13B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'llama-13b'
    
class Alpaca7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'alpaca-7b'
    
class Alpaca13B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'alpaca-13b'
    
class Wizard7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'wizard-7b'
    
class Vicuna7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'vicuna-7b'
    
class Chimera7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'chimera-7b'

class Baize7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = BloomForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'baize-7b'
    
class Guanaco7B(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = BloomForCausalLM.from_pretrained(self.model_path)
        return tokenizer,model
    @property
    def name(self):
        return 'guanaco-7b'
    
class TinyLlama(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,LlamaForCausalLM
        logging.debug('Loading tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = '<s>'
        logging.debug('Loading model')
        model = LlamaForCausalLM.from_pretrained(self.model_path)
        logging.debug('Done')
        return tokenizer,model
    @property
    def name(self):
        return 'tiny-llama'
    
def get_hf_model_classes():
    return [Llama7B,Llama13B,Alpaca7B,Alpaca13B,Wizard7B,Vicuna7B,Chimera7B,Baize7B,Guanaco7B,TinyLlama]
