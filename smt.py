import argparse
import torch
import random
import transformers
from transformers import AutoTokenizer, AutoModel
from DeBerta_model.model import BertCNN as bertmodel
#from DeBerta_model.model import LinearBert as bertmodel
#from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
import numpy as np
from torch.nn import BCEWithLogitsLoss, BCELoss
import logging
import re
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
import pandas
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

parser = argparse.ArgumentParser(description='Inference HateSpeech')

parser.add_argument('--path', default='data/input.txt', type=str, metavar='request_model', help='Model type to use, default: RNN')

path = "/home/nghianv/NLP/HSD/hatespeech-detection/deberta"
tokenizer = AutoTokenizer.from_pretrained(path)
print(tokenizer)
num_labels = 7
model = bertmodel(num_labels,path,max_len=384)
model.load_state_dict(torch.load("/home/nghianv/NLP/HSD/hatespeech-detection/save/model_3.pt"))

def padding(sequence, pads=0, max_len=None, dtype='int32',attention_mask = None, token_type_ids = None, global_attention_mask = None,padding_int=True):
    
    v_length = []
    for s in sequence:
        
        v_length+=[len(s)]# every sequence length
    
    seq_max_len = max(v_length)
    def round_to_power(x):
        return int((x-1)/512 + 1)*512

    if (max_len is None) or (max_len > seq_max_len) :
        if padding_int:
            max_len = round_to_power(seq_max_len) if max_len is None else min(max_len,round_to_power(seq_max_len))
        else:
            max_len = seq_max_len
    
    x = (np.ones((len(sequence),max_len)) * pads).astype(dtype)
    x_attention_mask = (np.ones((len(sequence),max_len)) * 0.).astype(dtype) if attention_mask is not None else None
    x_token_type_ids = (np.ones((len(sequence),max_len)) * 0.).astype(dtype) if token_type_ids is not None else None
    x_global_attention_mask = (np.ones((len(sequence), max_len)) * 0.).astype(dtype) if global_attention_mask is not None else None
    
    for idx, s in enumerate(sequence):
        
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
        if attention_mask is not None:
            x_attention_mask[idx,:len(trunc)] = attention_mask[idx][:max_len] 
        if token_type_ids is not None:
            x_token_type_ids[idx,:len(trunc)] = token_type_ids[idx][:max_len] 
        if global_attention_mask is not None:
            x_global_attention_mask[idx,:len(trunc)] = global_attention_mask[idx][:max_len]
    return x, x_attention_mask, x_token_type_ids



if __name__ == "__main__":
    global args
    args = parser.parse_args()
    model = model.to(device)
    with open(args.path,"r") as f:
        lines = f.readlines()
        for line in lines:
    #print("Input sentence: ")
            text = line.strip()
            encoded = tokenizer.encode_plus(text)
            input_ids = [encoded.input_ids]
            token_type_ids = [encoded.token_type_ids]
            attention_mask = [encoded.attention_mask]
            input_ids, attention_mask,token_type_ids = padding(input_ids, pads=tokenizer.pad_token_id, max_len=256,attention_mask = attention_mask, token_type_ids = token_type_ids)
            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device) if attention_mask is not None else None
            token_type_ids = torch.LongTensor(token_type_ids).to(device) if token_type_ids is not None else None
            out =model(input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
            #pred = (out>0.5).long()
            print(out)
