import pandas as pd
import numpy as np
import torch

from underthesea import word_tokenize
from transformers import RobertaTokenizer, BertConfig, AutoTokenizer, RobertaConfig, AutoConfig
from torch.utils.data import DataLoader, Dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

mapping = {0:"Toxicity",1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}

class ToxicityDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        data = convert_samples(
            text = self.text[item], 
            tokenizer = self.tokenizer,
            max_len = self.max_len,
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
        }

def convert_samples(text, tokenizer, max_len, labels=None):

    input_ids = tokenizer.encode_plus(text, padding='max_length', max_length=max_len, truncation=False)
    if len(input_ids['input_ids']) > max_len:
        input_ids_orig = input_ids['input_ids'][:150] + input_ids['input_ids'][-(max_len - 150):]
        attention_masks = input_ids['attention_mask'][:150] + input_ids['attention_mask'][-(max_len - 150):]
        token_type_ids = input_ids['token_type_ids'][:150] + input_ids['token_type_ids'][-(max_len - 150):]
    else:
        input_ids_orig = input_ids['input_ids']
        attention_masks = input_ids['attention_mask']
        token_type_ids = input_ids['token_type_ids']

    if labels is not None:
        return {
                'ids': input_ids_orig,
                'mask': attention_masks,
                'token_type_ids': token_type_ids,
                'label': label,
                }
    return  {
            'ids': input_ids_orig,
            'mask': attention_masks,
            'token_type_ids': token_type_ids,
            }

def convert_samples_to_ids(texts, tokenizer, max_seq_length, labels=None):
    input_ids, attention_masks = [], []

    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        masks = inputs['attention_mask']
        attention_masks.append(masks)

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)


def predict_phoBERT(sentence, tokenizer, model, max_seq_len=256, device=device):

    test_set    = ToxicityDataset([sentence], tokenizer, max_seq_len)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False)

    tk0 = (enumerate(test_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            input_ids, input_masks, input_segments = batch['ids'], batch['mask'], batch['token_type_ids']
            y_pred = model(input_ids.to(device), input_masks.to(device), token_type_ids=input_segments)
            y_pred = y_pred.squeeze().detach().cpu().numpy()

    return y_pred


def predict_file_phoBERT(file, tokenizer, model, batch_size=16, max_len=256):

    if 'csv' in file:
      test = pd.read_csv(file)
      if 'spoken' in file:
        sentences = test.normed_comments.values
      else:
        sentences = test.comments.values
    else:
      with open(file) as f:
        sentences = f.read_lines()
        sentences = [i.strip() for i in sentences]

    # sentences = sentences[:30]
    test_set    = ToxicityDataset(sentences, tokenizer, max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    test_shape = len(sentences)
    test_preds = np.zeros((test_shape, 7))
    
    tk0 = (enumerate(test_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            input_ids, input_masks, input_segments = batch['ids'], batch['mask'], batch['token_type_ids']
            input_ids, input_masks, input_segments = input_ids.to(device), input_masks.to(device), input_segments.to(device)
            
            logits = model(input_ids = input_ids,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
            
            test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
        preds = torch.sigmoid(torch.tensor(test_preds)).numpy()
        output = np.zeros((len(x_test), 8))
        output[:len(x_test),:7] = preds
        output[:,7] = 1 - output[:,0]
        output = list(np.argmax(output, axis = 1))
        output = [mapping[i] for i in output]
        
    return output, preds


def predict_sample_VA_phoBERT(file, tokenizer, model, batch_size=16, max_len=120):

    data = pd.read_excel(file)
    sentences = data.Sample.astype(str).values
    
    test_set    = ToxicityDataset(sentences, tokenizer, max_len)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
    test_shape = len(sentences)
    test_preds = np.zeros((test_shape, 7))
    
    tk0 = (enumerate(test_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            input_ids, input_masks, input_segments = batch['ids'], batch['mask'], batch['token_type_ids']
            input_ids, input_masks, input_segments = input_ids.to(device), input_masks.to(device), input_segments.to(device)
            
            logits = model(input_ids = input_ids,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
            
            test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
        test_preds = torch.sigmoid(torch.tensor(test_preds)).numpy()
        outputs = []
        x = test_preds[:,0] > 0.5

        for index, value in enumerate(x):
            output = {}
            if value == True:
                max_indice = list(test_preds[index, 1:]).index(max(test_preds[index, 1:]))
                output['Category'] = mapping[max_indice]
                output['Toxicity'] = 1
                outputs.append(output)
            else:
                outputs.append('Non-Toxic')
        
    return outputs, test_preds