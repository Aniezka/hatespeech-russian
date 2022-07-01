import pandas as pd 
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-conversational')
model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-conversational', output_hidden_states=True).eval()

def find_cosin_similarity(text_1, text_2, text_3):
    tok1 = tokenizer(text_1, return_tensors='pt')
    tok2 = tokenizer(text_2, return_tensors='pt')
    tok3 = tokenizer(text_3, return_tensors='pt')

    with torch.no_grad():
        out1 = model(tok1.input_ids)
        out2 = model(tok2.input_ids)
        out3 = model(tok3.input_ids)
    
    # Only grab the last hidden state
    states1 = out1.last_hidden_state.squeeze()
    states2 = out2.last_hidden_state.squeeze()
    states3 = out3.last_hidden_state.squeeze()
    
    # average words vectors
    avg1 = states1.mean(axis=0)
    avg2 = states2.mean(axis=0)
    avg3 = states3.mean(axis=0)
    
    cosin_origin = torch.cosine_similarity(avg1.reshape(1,-1), avg2.reshape(1,-1))
    cosin_non_toxic = torch.cosine_similarity(avg1.reshape(1,-1), avg3.reshape(1,-1))
    if cosin_origin < cosin_non_toxic: 
        who_won = 'toxic_original'
    else: 
        who_won = 'non_toxic_ipa'

    return cosin_origin, cosin_non_toxic, who_won



data = pd.read_csv('gpt_rubert_data.csv')
replica_1 = list(data["Реплика 1 toxicity"])
replica_2_toxic_original = list(data["Реплика 2 toxicity"])
replica_2_nontoxic_ipa_1 = list(data["Реплика 2 original dialogue 1"])

cosins_origins_toxic = []
cosins_non_toxics_ipa = []
who_wins = []
for text_1, text_2, text_3 in tqdm(zip(replica_1, replica_2_toxic_original, replica_2_nontoxic_ipa_1)):
    cosin_origin, cosin_non_toxic, who_won = find_cosin_similarity(text_1, text_2, text_3)
    cosins_origins_toxic.append(cosin_origin)
    cosins_non_toxics_ipa.append(cosin_non_toxic)
    who_wins.append(who_won)


data['RubertXL choice'] = who_wins 
data.to_csv('gpt_rubert_ruxl_data.csv')