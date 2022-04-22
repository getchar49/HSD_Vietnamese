import numpy as numpy
import pandas as pd 
from transformers import BertPreTrainedModel, AutoModel
import torch
import torch.nn as nn 
import torch.nn.functional as F

########## Model Bert+CNN+label attention ######
# get 4 hidden state in transformer + CNN, concat label attention in after CNN
class BertCNN(nn.Module):
    def __init__(self, num_labels, MODEL_NAME, lb_availible=False, max_len=256, max_len_tfidf=0):
        #super(BertCNN, self).__init__(conf)
        super().__init__()
        self.lb_availible = lb_availible
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)

        self.convs = nn.ModuleList([nn.Conv1d(max_len+max_len_tfidf, 256, kernel_size,padding='same') for kernel_size in [3,5,7]])
        #self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1152, self.num_labels)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, hidden_outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = False,
            output_hidden_states=True
        )
       # print(sequence_output)
        #print(hidden_outputs[-1].shape)
        sequence_output = torch.stack([hidden_outputs[-1], hidden_outputs[-2], hidden_outputs[-3]])
        sequence_output = torch.mean(sequence_output, dim=0)
        cls_output = sequence_output[:,0,:]
        sequence_output = sequence_output[:,1:,:]
        sequence_output = torch.transpose(sequence_output,1,2)
        #print("Sequence shape: ",sequence_output.shape)
        #print(sequence_output.shape)
        cnn = [F.relu(conv(sequence_output)) for conv in self.convs]
        max_pooling = []
        max_pooling.append(cls_output)
        for i in cnn:
          #print(i.shape)
          max, _ = torch.max(i, 2)
          #print(max.shape)
          max_pooling.append(max)
        output = torch.cat(max_pooling, 1)
        #print(output.shape)
        output = self.dropout(output)
        logits = self.out(output)
        #print(torch.sigmoid(logits))
        #return logits
        return torch.clamp(torch.sigmoid(logits),1e-7,1-1e-7)

class LinearBert(nn.Module):
    def __init__(self, num_labels, MODEL_NAME, lb_availible=False, max_len=256, max_len_tfidf=0):
        #super(BertCNN, self).__init__(conf)
        super().__init__()
        self.lb_availible = lb_availible
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)

        self.convs = nn.ModuleList([nn.Conv1d(max_len+max_len_tfidf, 256, kernel_size,padding='same') for kernel_size in [3,5,7]])
        #self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(384, self.num_labels)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, hidden_outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = False,
            output_hidden_states=True
        )
        #print(sequence_output)
        #print(hidden_outputs[-1])
        #print(hidden_outputs[-1].shape)
        #sequence_output = torch.stack([hidden_outputs[-1], hidden_outputs[-2], hidden_outputs[-3]])
        #sequence_output = torch.mean(sequence_output, dim=0)
        #sequence_output = torch.transpose(sequence_output,1,2)
        #print("Sequence shape: ",sequence_output.shape)
        #print(sequence_output.shape)
        #cnn = [F.relu(conv(sequence_output)) for conv in self.convs]
        #max_pooling = []
        #self.out(output)
        #print(torch.sigmoid(logits))
        #return logits
        #print(sequence_output[:,0,:].shape)
        #print(sequence_output[:,0,:])
        logits = self.out(sequence_output[:,0,:])
        return torch.clamp(torch.sigmoid(logits),1e-7,1-1e-7)

class BertBase(nn.Module):
    def __init__(self):
    
        #super(BertBase, self).__init__(conf)
        super().__init__()
        #self.config = conf
        self.backbone = XLMRobertaModel.from_pretrained(
            model_name)
        self.lstm_units = 768
        self.num_recurrent_layers = 1
        self.bidirectional = False

        self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                            hidden_size=self.lstm_units,
                            num_layers=self.num_recurrent_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(self.config.hidden_size*2, 7)

    def forward(self, input_ids, attention_mask, token_type_ids):

        # with autocast():
        sequence_output, _, hidden_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

            # sequence_output = torch.stack(hidden_outputs[-2:]).mean(0)
            # sequence_output = torch.cat(tuple([hidden_outputs[i] for i in [-1, -2]]), dim=-1)

        if self.bidirectional:
            n = 2
        else: n = 1

        h0 = Variable(torch.zeros(self.num_recurrent_layers * n,       # (L * 2 OR L, B, H)
                                      input_ids.shape[0],
                                      self.lstm_units))
        c0 = Variable(torch.zeros(self.num_recurrent_layers * n,        # (L * 2 OR L, B, H)
                                      input_ids.shape[0],
                                      self.lstm_units))

        sequence_output, _ = self.lstm(sequence_output, (h0, c0))

        avg_pool = torch.mean(sequence_output, 1)
        max_pool, _ = torch.max(sequence_output, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        output = self.dropout(h_conc)
        logits = self.out(output)

        return logits