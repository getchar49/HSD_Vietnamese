# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Vietnamese HateSpeech-Toxicity Detection ###

Hate speech and toxicity detection for vietnamese language (spoken form). Data preparation, modelling and results are reported here:
 
https://vinbdi-slp.atlassian.net/wiki/spaces/NLP/pages/337641519/Hate+Speech+-+Toxicity+in+Vietnamese+Language

### INFERENCE ###

##### To run model inference on a command as a service, execute the following: #####

`python app.py`

`url = "http://0.0.0.0:5000/result?format=json"`

`requests.post(url, data={'result':'xin chao', 'model':'RNN'}).json()`

result: command to be inferred

model: RNN, enviBERT, phoBERT


##### To run RNN model inference on VA's general commands, execute the following: #####

`python inference.py --request_model "RNN" --weight_file "path/to/model_weight" --sample_VA "path/to/sample_intent.xlsx"`

##### To run enviBERT model inference on VA's general commands, execute the following: #####

`python inference.py --request_model "enviBERT" --model_folder "path/to/enviBERT_pretrained" --weight_file "path/to/model_weight" --sample_VA "path/to/sample_intent.xlsx"`

request_model: RNN, enviBERT, phoBERT

models_folder: path/to/test_set.csv


* Folder structure

```text
EnviBERT_model
	|---- enviBERT
        |----- dict.txt
        |----- sentencepiece.bpe.model
	|---- weights
		|----- spoken_form_phoBert_model_v4.pt

RNN_model
	|---- accessories
		|----- vectors_200d_v1.txt
	|---- weights
		|----- spoken_form_RNN_model_200d_v4.pt
```

weight_file:

	_ RNN model: path/to/RNN_model

	_ enviBERT: path/to/enviBERT_model

	_ phoBERT: path/to/phoBERT_model

models_folder:

	_ enviBERT: path/to/pretrained_enviBERT

	_ phoBERT: 'vinai/phobert-base'


## Note about files ##

Weights can be found at: 10.124.68.101/slp/09_pretrained/hatespeech/

All the weights are currently at version 4 with "v4" in their names, except for phoBert which is v3 and not included in deployment. 

Data can be found at: 10.124.68.101/slp/04_databases/33_vn_hatespeech/

RNN accessory files can be found at: 10.124.68.101/slp/04_databases/33_vn_hatespeech/RNN_accessories

	For 2 folders in this data directory, please leave Datasets folder on the same level of the project, while accessory files for RNN should be in the RNN_model/accessories folder
