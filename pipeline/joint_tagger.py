import argparse

import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm

from data_util import *
from util import *
from paragraph_model import JointParagraphTagger
from dataset import JointPredictionDataset

import logging

import pickle

#class dummyArgs():
#    def __init__(self):
#        pass
    
#    def set_args(self, args_dict):
#        for k, v in args_dict.items():
#            setattr(self, k, v)

#args = dummyArgs()
            
def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
def index2label(all_indices, mapping):
    all_labels = []
    for indices in all_indices:
        all_labels.append([mapping.get(index,"pad") for index in indices])
    return all_labels

def predict(model, dataset, tokenizer, device, args):
    model.eval()
    discourse_predictions = []
    citation_predictions = []
    span_predictions = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            try:
                encoded_dict = encode(tokenizer, batch, has_local_attention="led-" in args.repfile or "longformer" in args.repfile or "led_" in args.repfile)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"],
                                                               tokenizer.sep_token_id)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                discourse_out, citation_out, span_out, _, _, _ = model(encoded_dict, transformation_indices, batch["N_tokens"])
                discourse_predictions.extend(index2label(discourse_out, dataset.discourse_label_lookup))
                citation_predictions.extend(index2label(citation_out, dataset.citation_label_lookup))
                span_predictions.extend(index2label(span_out, dataset.span_label_lookup))
            except:
                print(batch)
    return discourse_predictions, citation_predictions, span_predictions


def encode(tokenizer, batch, has_local_attention=False):
    inputs = batch["paragraph"]
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True, add_special_tokens=True,
        return_tensors='pt')
    if has_local_attention:
        # additional_special_tokens_lookup = {token: idx for token, idx in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids)}
        # special_token_ids = set([additional_special_tokens_lookup[token] for token in special_tokens])
        # special_token_ids.add(tokenizer.mask_token_id)
        special_token_ids = tokenizer.additional_special_tokens_ids
        special_token_ids.append(tokenizer.sep_token_id)

        batch_size, MAX_SENT_LEN = encoded_dict["input_ids"].shape
        global_attention_mask = batch_size * [
            [0 for _ in range(MAX_SENT_LEN)]
        ]
        for i_batch in range(batch_size):
            for i_token in range(MAX_SENT_LEN):
                if encoded_dict["input_ids"][i_batch][
                    i_token] in special_token_ids:
                    global_attention_mask[i_batch][i_token] = 1
        encoded_dict["global_attention_mask"] = torch.tensor(
            global_attention_mask)
    # Single pass to BERT should not exceed max_sent_len anymore, because it's handled in dataset.py
    return encoded_dict


def token_idx_by_sentence(input_ids, sep_token_id, padding_idx=-1):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,
                               1).numpy().tolist()  # Number of sentences per paragraph + 1
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(
        sep_tokens.size(0), -1)
    sep_indices = torch.split(indices[sep_tokens], paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        word_indices = [torch.arange(paragraph[i], paragraph[i + 1]) for i in
                        range(paragraph.size(0) - 1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)

    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices,
                                                    batch_first=True,
                                                    padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence, paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split,
                                                 batch_first=True,
                                                 padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(
        -1).expand(-1, indices_by_batch.size(1), indices_by_batch.size(-1))
    mask = (indices_by_batch != padding_idx)
    return batch_indices.long(), indices_by_batch.long(), mask.long()

def run_prediction(input_dict, tokenizer, args):
    #args.set_args(arg_dict)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    #additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    #tokenizer.add_special_tokens(additional_special_tokens)
    
    dev_set = JointPredictionDataset(input_dict, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)

    model = JointParagraphTagger(args.repfile, len(tokenizer),args.dropout)#.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    print("Model loaded!")
    
    model = model.to(device)
    discourse_predictions, citation_predictions, span_predictions = predict(model, dev_set, tokenizer, device, args)
    citation_predictions = fix_BIO(citation_predictions)
    span_predictions = fix_BIO(span_predictions)
    span_predictions = post_process_spans(span_predictions, citation_predictions)
    
    #with open("discourse_predictions.pkl","wb") as f:
    #    pickle.dump(discourse_predictions, f)
    #with open("citation_predictions.pkl","wb") as f:
    #    pickle.dump(citation_predictions, f)
    #with open("span_predictions.pkl","wb") as f:
    #    pickle.dump(span_predictions, f)
    #with open("dataset.pkl","wb") as f:
    #    pickle.dump(dev_set, f)
    return discourse_predictions, citation_predictions, span_predictions, dev_set