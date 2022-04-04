import torch
import numpy as np
import pandas as pd
from RNN_model.preprocessing import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset
from RNN_model.models import *
from RNN_model.tokenizer import ToxicityDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

file = open('RNN_model/accessories/full_comments_v4.txt','r')
docs = file.readlines()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)


mapping = {0:"Toxicity",1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_RNN(sentence, model):
   sentence = preprocess(sentence)
   x_infer = tokenizer.texts_to_sequences([sentence])
   x_infer = pad_sequences(x_infer, maxlen=400)
   infer = ToxicityDataset(x_infer)
   infer_loader = DataLoader(infer, batch_size=1, shuffle=False)
   test_preds = np.zeros((len(infer), 7))
   for i, x_batch in enumerate(infer_loader):
      text = x_batch['text'].to("cpu")
      y_pred = sigmoid(model(text).detach().cpu().numpy())
      test_preds[i * 1:(i+1) * 1, :] = y_pred

   return test_preds[0]

def predict_file_RNN(file, model, batch_size=64):

   if 'csv' in file:
      test = pd.read_csv(file)
      if 'spoken' in file:
        sentences = test.normed_comments.map(lambda x: preprocess(x))
      else:
        sentences = test.comments.map(lambda x: preprocess(x))
   else:
      with open(file) as f:
        sentences = f.read_lines()
        sentences = [i.strip() for i in sentences]

   x_test = tokenizer.texts_to_sequences(sentences)
   x_test = pad_sequences(x_test, maxlen=300)

   test_set    = ToxicityDataset(x_test)
   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
   test_shape = len(x_test)
   test_preds = np.zeros((test_shape, 7))
    
   tk0 = enumerate(test_loader)
   with torch.no_grad():
        
      for idx, batch in tk0:
         text = batch['text'].to(device)
            
         logits = model(text)
         test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
      output = np.zeros((len(x_test), 8))
      output[:len(x_test),:7] = preds
      output[:,7] = 1 - output[:,0]
      output = list(np.argmax(output, axis = 1))
      output = [mapping[i] for i in output]
        
   return output, preds


def predict_sample_VA_RNN(file, model, batch_size=64):

   data = pd.read_excel(file)
   sentences = data.Sample.astype(str).map(lambda x: preprocess(x))

   x_test = tokenizer.texts_to_sequences(sentences)
   x_test = pad_sequences(x_test, maxlen=120)

   test_set    = ToxicityDataset(x_test)
   test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
    
   test_shape = len(x_test)
   test_preds = np.zeros((test_shape, 7))
    
   tk0 = enumerate(test_loader)
   with torch.no_grad():
        
      for idx, batch in tk0:
         text = batch['text'].to(device)
            
         logits = model(text)
         test_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
        
      preds = torch.sigmoid(torch.tensor(test_preds)).numpy()

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