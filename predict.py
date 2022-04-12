import argparse
import torch
import random
import transformers
from transformers import AutoTokenizer, AutoModel
from DeBerta_model.model import BertCNN as bertmodel
#from DeBerta_model.model import LinearBert as bertmodel
from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
import numpy as np
from torch.nn import BCEWithLogitsLoss, BCELoss
import logging
import re
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
import pandas
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

path = "/content/HSD_Vietnamese/deberta"
save_path = "/content/drive/MyDrive/Colab Notebooks/HSD/save"
data_path = "/content/drive/MyDrive/Colab Notebooks/HSD/data"

df = pandas.read_csv(data_path+"/test_v9.csv")
num_labels = 7
model = bertmodel(num_labels,path,max_len=384)
layers = model.backbone.config.num_hidden_layers
#model = AutoModel.from_pretrained(path)
#print(model)
tokenizer = AutoTokenizer.from_pretrained(path)
model.load_state_dict(torch.load(open(save_path+'/model_0.pt',"rb")))
total_data = []
#print(dir(df))
#print(df.shape)
#print(df.iloc[df.size-1,:])
#print(df.shape)
#arr = np.zeros(1500)
cout = torch.zeros(7) * 1.0
for i in range(df.shape[0]):
    row_i = df.iloc[i,:].to_list()
    text = row_i[0]
    label = row_i[1:]
    cout += torch.FloatTensor(label)
    #print(dir(tokenizer))
    encoded = tokenizer.encode_plus(text)
    # if (len(encoded)==1242):
    #     print(text)
    #arr[len(encoded)] += 1
    #max_len = max(max_len,len(encoded))
    total_data.append({
        'input_ids':encoded.input_ids,
        'token_type_ids':encoded.token_type_ids,
        'attention_mask':encoded.attention_mask,
        'label':label
    })
    
test_data = total_data
total_test = len(test_data)
#print
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

with torch.no_grad():
    result = torch.zeros((7,2,2))
    batch_size = 4
    model = model.to(device)
    #model = model.cpu()
    for i in range(0,total_test,batch_size):
        data = test_data[i:i+batch_size]
        input_ids = [d['input_ids'] for d in data]
        #print(tokenizer.decode(data[0]['input_ids']))
        #print(data[0]['label'])
        #print(input_ids.shape)
        token_type_ids = [d['token_type_ids'] for d in data]
        attention_mask = [d['attention_mask'] for d in data]
        label = torch.LongTensor([d['label'] for d in data])
        input_ids, attention_mask,token_type_ids = padding(input_ids, pads=tokenizer.pad_token_id, max_len=256,attention_mask = attention_mask, token_type_ids = token_type_ids)
        #print(data)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device) if attention_mask is not None else None
        token_type_ids = torch.LongTensor(token_type_ids).to(device) if token_type_ids is not None else None
        out =model(input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
        #print("Output: ",out)
        #print("Label: ",label)
        #if(i%10==0):
        #    print(label)
        #    print(out)
        pred = (out>0.5).long().cpu()
        
        result += multilabel_confusion_matrix(label,pred)
    f1_scores = torch.zeros(result.shape[0])
    for i in range(result.shape[0]):
      f1_scores[i] = result[i][1][1]*2/(result[i][1][1]*2+result[i][0][1]+result[i][1][0])
    print(result)
    print(f1_scores)