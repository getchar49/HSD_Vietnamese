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
#record = {'loss':[],'avg_loss':[],'val_acc':[],'optim':None,'sche':None,'epoch':None,'total_loss':None,'best_acc':0.24}

params = {
    'dropout': {0,0.1,0.15},
    'warmup': {50,100,500,1000},
    'lr': {1.5e-5,2e-5, 2.5e-5, 3e-5},
    'batch_size': {16,32,48,64}
}

torch.manual_seed(0)
np.random.seed(0)

#path = "/home/nghianv/NLP/HSD/hatespeech-detection/deberta"
path = "/content/HSD_Vietnamese/deberta"
save_path = "/content/drive/MyDrive/Colab Notebooks/HSD/save"
data_path = "/content/drive/MyDrive/Colab Notebooks/HSD/data"
num_labels = 7
model = bertmodel(num_labels,path,max_len=384)
layers = model.backbone.config.num_hidden_layers
#model = AutoModel.from_pretrained(path)
#print(model)
tokenizer = AutoTokenizer.from_pretrained(path)
#tokenizer = DebertaV2Tokenizer.from_pretrained(path)
#tokenizer = AutoTokenizer.from_pretrained(path)
#print(tokenizer)



record = {'loss':[],'avg_loss':[],'val_acc':[],'optim':None,'sche':None,'epoch':None,'total_loss':0,'best_acc':0,'score':[]}

df = pandas.read_csv(data_path+"/train_v9.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--split_layer", type=int, default=3)
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default = 0.1
)
parser.add_argument(
    "--alpha",
    type=float,
    default = 5
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
args = parser.parse_args()
#logging.info(args)


#print(df.columns)

#labels = ['Toxicity', 'Obscence', 'Threat', 'Identity attack - Insult', 'Sexual explicit', 'Sedition – Politics', 'Spam']

st = "I go to school"

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lambda1 = (lambda epoch:float(epoch) / float(max(1, num_warmup_steps)) if epoch < num_warmup_steps \
                else max(0.0, float(num_training_steps - epoch) / float(max(1, num_training_steps - num_warmup_steps))))
    
    return LambdaLR(optimizer, lambda1, last_epoch)

def optim(n_layers,lr,model,alpha=5.,split_layers=12):
    params = dict(model.named_parameters())

    layer_params =  [] 
    for _ in range(n_layers):
        layer_params.append([])
    
    layer_params[0].extend(list(model.backbone.embeddings.parameters()))
    for key,value in params.items():
        #print(key)
        layer = re.findall(r"backbone\.encoder\.layer\.(.*?)\.",key)
        if len(layer) == 1:    
            exec("layer_params[{}].append(value)".format(int(layer[0])))

    for key,value in params.items():
        if ("backbone.encoder.rel_embeddings." in key):
          layer_params[-1].append(value)
        if ("backbone.encoder.LayerNorm." in key):
          layer_params[-1].append(value)
        if "convs." in key:
            layer_params[-1].append(value)
        if "out." in key:
            layer_params[-1].append(value)
    #layer_params[-1].extend(list(model.backbone.pooler.parameters()))
   
    length = [len(p) for p in layer_params]
    assert sum(length) == len(params)

    params_list = []
    one_layer_block_num = n_layers/split_layers 
    for i,layer_param in enumerate(layer_params):
        m = alpha ** (split_layers-i//one_layer_block_num-1)
        params_list.append({'params':layer_param,'lr':lr/m})
    #print(params_list)
    optimizer2 = torch.optim.AdamW(params_list,
                                  weight_decay=0.01)
    #print(optimizer2)
    return optimizer2 

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
    
def optim(n_layers,lr,model,alpha=5.,split_layers=12):
    params = dict(model.named_parameters())
    layer_params =  [] 
    for _ in range(n_layers):
        layer_params.append([])
    
    layer_params[0].extend(list(model.encoder.embeddings.parameters()))
    for key,value in params.items():
        layer = re.findall(r"encoder\.encoder\.layer\.(.*?)\.",key)
        if len(layer) == 1:    
            exec("layer_params[{}].append(value)".format(int(layer[0])))

    for key,value in params.items():
        if "predictions." in key:
            layer_params[-1].append(value)
        if "denses." in key:
            layer_params[-1].append(value)
    layer_params[-1].extend(list(model.encoder.pooler.parameters()))
   
    length = [len(p) for p in layer_params]
    assert sum(length) == len(params)

    params_list = []
    one_layer_block_num = n_layers/split_layers 
    for i,layer_param in enumerate(layer_params):
        m = alpha ** (split_layers-i//one_layer_block_num-1)
        params_list.append({'params':layer_param,'lr':lr/m})
    #print(params_list)
    optimizer2 = torch.optim.AdamW(params_list,
                                  weight_decay=0.01)
    #print(optimizer2)
    return optimizer2 

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
#print(cout)
class_weight = cout / df.shape[0]
class_weight = (1-class_weight)/class_weight
class_weight = (torch.ones(7))*1
scale = ((class_weight**2+1)/2)**0.5

scale = scale.requires_grad_(False).to(device)
class_weight = class_weight.requires_grad_(False).to(device)
#print(scale)
#class_weight = (torch.ones(7)*4).to(device)
print(class_weight)
#random.shuffle(total_data)
print("Length total data: ",len(total_data))
val_data = total_data[:len(total_data)//10]
train_data = total_data[len(total_data)//10:]
#data = train_data[0:200]
#optimizer = optim(layers,args.lr,model,args.alpha,split_layers=args.split_layer)
total_steps = len(train_data) * args.epoch // (args.gradient_accumulation_steps * args.batch_size)
#scheduler = get_linear_schedule_with_warmup(optimizer,int(args.warmup_proportion*total_steps),total_steps)

#device = 'cpu'
batch_size = 16
total_loss = 0.
cur_loss = 0.
step = 0
cur_epo = 0
#memory = [0]
model = model.to(device)
#total = 5000
total = len(train_data)
print("Total: ",total)
total_val = len(val_data)

optimizer = transformers.AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer,500,total_steps)
#scheduler = get_linear_schedule_with_warmup(optimizer,int(args.warmup_proportion*total_steps),total_steps)
criterion = BCEWithLogitsLoss(pos_weight = class_weight)
def loss_function(x,y):
    #print((class_weight*y*torch.log(x) + (1-y)*torch.log(1-x))/scale)
    return -torch.mean((class_weight*y*torch.log(x) + (1-y)*torch.log(1-x))/scale)
sigmoid = torch.nn.Sigmoid()
max_f1_score = 0.
for epo in range(cur_epo,args.epoch):
  model = model.to(device)
  batch_size = 16
  prev_total_loss = (0,0)
  model.train()
  random.shuffle(train_data)
  for i in range(0,total,batch_size):
    data = train_data[i:i+batch_size]
    input_ids = [d['input_ids'] for d in data]
    #print(tokenizer.decode(data[0]['input_ids']))
    #print(data[0]['label'])
    #print(input_ids.shape)
    token_type_ids = [d['token_type_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]
    label = torch.FloatTensor([d['label'] for d in data])
    input_ids, attention_mask,token_type_ids = padding(input_ids, pads=tokenizer.pad_token_id, max_len=256,attention_mask = attention_mask, token_type_ids = token_type_ids)
    #print(data)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device) if attention_mask is not None else None
    token_type_ids = torch.LongTensor(token_type_ids).to(device) if token_type_ids is not None else None
    label = label.to(device) if label is not None else None
    #print(input_ids.shape)
    out =model(input_ids,attention_mask = attention_mask,token_type_ids = token_type_ids)
    #print(label)
    #print(out)
    #print(out[0].shape)
    #out = torch.clamp(out,-100,100)
    # if (i%500==0):
    #     print(i)
    #     print(out)
    #     print(label)
        #print(scale)
        #print(class_weight)
    
    #loss = loss_function(out,label)
    loss = loss_function(out,label)
    #loss = criterion(out,label)
    #loss = loss/scale
    #print(loss)
    #print(loss)
    loss.backward()
    if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.warmup_proportion is not None and args.warmup_proportion!=0.0:
                scheduler.step()
    step += 1
    #print(loss)
    total_loss += float(loss.cpu())
    #print(total_loss)
    avg_loss = total_loss/step
    #total_losses.append(total_loss)
    #print(total_loss)
    x = 50
    if (step%x==0): 
      #print(step)
      #x = memory[-1]  
      #print(total_losses)  
      #print(total_losses)
      print("Step {}/{}: Avg loss: {} Cur avg loss: {}".format(step*batch_size,total,avg_loss,(total_loss-prev_total_loss[0])/x))
      #print(total_loss,prev_total_loss)
      #memory.append(total_loss)
      prev_total_loss = (total_loss,step*batch_size)
      #memory = memory[-5:]
  print("Step {}/{}: Avg loss: {} Cur avg loss: {}".format(total,total,total_loss/((total+1)//batch_size),(total_loss-prev_total_loss[0])/((total-prev_total_loss[1]+1)//batch_size)))
  step = 0
  total_loss = 0
  prev_total_loss = 0
  model.eval()
  result = torch.zeros((7,2,2))
  batch_size = 2
  with torch.no_grad():
    #model = model.cpu()
    for i in range(0,total_val,batch_size):
        data = val_data[i:i+batch_size]
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
  print("Epoch {} result:".format(epo+1))
  print(result.long())
  print(f1_scores)
  with open(save_path+'/model_{}.pt'.format(0), 'wb') as f:
    state_dict = model.state_dict()
    torch.save(state_dict, f)
  avg_f1_score = np.mean(f1_scores)
  if (avg_f1_score>max_f1_score):
    max_f1_score = avg_f1_score
    with open(save_path+'/model_max.pt', 'wb') as f:
      state_dict = model.state_dict()
      torch.save(state_dict, f)
  #record['optim'] = optimizer.state_dict()
  #record['sche'] = scheduler.state_dict()
  record['score'].append(f1_scores)
  with open(save_path+'/record_{}.pt'.format(0),'wb') as f:
    torch.save(record,f)
#print(criterion(out,label))
for i in range(len(train_data)):
