import time
import json
import torch
import pandas as pd
import torch
import numpy as np
import os
import argparse
from transformers import RobertaTokenizer, BertConfig, AutoTokenizer, RobertaConfig, AutoConfig
from PhoBERT_model.models import *
from PhoBERT_model.utils import predict_phoBERT, predict_file_phoBERT, predict_sample_VA_phoBERT
from EnviBERT_model.tokenizer import XLMRobertaTokenizer
from EnviBERT_model.utils import predict_enviBERT, predict_file_enviBERT, predict_sample_VA_enviBERT
from RNN_model.models import *
from RNN_model.utils import predict_RNN, predict_file_RNN, predict_sample_VA_RNN
from Norm_TTS import norm_sentence


parser = argparse.ArgumentParser(description='Inference HateSpeech')
# Directory where we want to write everything we save in this script to
parser.add_argument('--request_model', default='enviBERT', type=str, metavar='request_model', help='Model type to use, default: RNN')
parser.add_argument('--data_folder', default='Datasets/test_v8.csv', metavar='DIR',
                    help='folder to retrieve data, text files, etc.')
parser.add_argument('--models_folder', default='EnviBERT_model/enviBERT', metavar='DIR',
                    help='folder to load models')
parser.add_argument('--weight_file', default='EnviBERT_model/weights/spoken_form_EnviBERT_model_v4.pt', metavar='DIR',
                    help='weight of model')
parser.add_argument('--sample_VA', default='../HateSpeech/data/VF_VA_sample_intent.xlsx', metavar='DIR',
                    help='VA samples to test')

# phoBERT = 'vinai/phobert-base'
# 'PhoBERT_model/weights/spoken_form_phoBert_model_v4.pt'

# enviBERT = 'EnviBERT_model/enviBERT'
# 'EnviBERT_model/weights/spoken_form_EnviBERT_model_v4.pt'

# RNN_model/weights/spoken_form_RNN_model_200d_v4.pt
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
   
if __name__ == '__main__':

   # Parse commands from ArgumentParser
   global args
   args = parser.parse_args()

   # sentence = 'con chó này'

   # if args.request_model == 'enviBERT':
   #    tokenizer = XLMRobertaTokenizer(args.models_folder)
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    result = predict_enviBERT(sentence, tokenizer = tokenizer, model = model)
   # elif args.request_model == 'phoBERT':
   #    tokenizer = AutoTokenizer.from_pretrained(args.models_folder, use_fast=False)
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    result = predict_phoBERT(sentence, tokenizer = tokenizer, model = model)
   # else:
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    result = predict_RNN(sentence, model)

   # mapping = {1:"Obscence",2:"Threat",3:"Identity attack-Insult",4:"Sexual-explicit",5:"Sedition-Politics",6:"Spam"}
   
   # output = {}
   # result_list = list(result)
   # list_results = []

   # if result[0] > 0.5:
   #    max_indice = result_list.index(max(result_list[1:]))
   #    output['Category'] = mapping[max_indice]
   #    output['Toxicity'] = 1
   # else:
   #    output['Toxicity'] = 0
   # list_results.append(output)
   # print(list_results)

   # if args.request_model == 'enviBERT':
   #    tokenizer = XLMRobertaTokenizer(args.models_folder)
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    pred_target, probs = predict_file_enviBERT(args.data_folder, tokenizer, model)
   # elif args.request_model == 'phoBERT':
   #    tokenizer = AutoTokenizer.from_pretrained(args.models_folder, use_fast=False)
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    pred_target, probs = predict_file_phoBERT(args.data_folder, tokenizer, model)
   # else:
   #    model = torch.load(args.weight_file, map_location=device)
   #    model.eval()
   #    pred_target, probs = predict_file_RNN(args.data_folder, model)
    
   # print(pred_target)  

   if args.request_model == 'enviBERT':
      tokenizer = XLMRobertaTokenizer(args.models_folder)
      model = torch.load(args.weight_file, map_location=device)
      model.eval()
      pred_target, probs = predict_sample_VA_enviBERT(args.sample_VA, tokenizer, model)
   elif args.request_model == 'phoBERT':
      tokenizer = AutoTokenizer.from_pretrained(args.models_folder, use_fast=False)
      model = torch.load(args.weight_file, map_location=device)
      model.eval()
      pred_target, probs = predict_sample_VA_phoBERT(args.sample_VA, tokenizer, model)
   else:
      model = torch.load(args.weight_file, map_location=device)
      model.eval()
      pred_target, probs = predict_sample_VA_RNN(args.sample_VA, model)

   TP = pred_target.count("Non-Toxic")
   print("FP rate is: ", 1 - TP/len(pred_target))
   # print(pred_target)  