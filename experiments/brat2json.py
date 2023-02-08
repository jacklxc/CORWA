import argparse

import torch
import jsonlines
import os
import pickle
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import flatten, read_related_work_jsons, annotate_related_work
from paragraph_model import JointParagraphTagger
#from paragraph_model import JointParagraphCRFTagger as JointParagraphTagger 
from dataset import JointRelatedWorkAnnotationDataset

import logging


def batch_token_label(labels, padding_idx):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        label_indices = [int(evid) for evid in label]
        label_matrix[i,:len(label_indices)] = torch.tensor(label_indices)
        label_list.append(label_indices)
    return label_matrix.long(), label_list

def index2label(all_indices, mapping):
    all_labels = []
    for indices in all_indices:
        all_labels.append([mapping.get(index,"pad") for index in indices])
    return all_labels


def extract_label(dataset):
    discourse_labels = []
    citation_labels = []
    span_labels = []
        
    for batch in tqdm(DataLoader(dataset, batch_size = 1, shuffle=False)):
        padded_discourse_label, discourse_label = batch_token_label(batch["discourse_label"], 0) 
        padded_citation_label, citation_label = batch_token_label(batch["citation_label"], 0) 
        padded_span_label, span_label = batch_token_label(batch["span_label"], 0) #len(dev_set.span_label_types))

        discourse_labels.extend(index2label(discourse_label, dataset.discourse_label_lookup))
        citation_labels.extend(index2label(citation_label, dataset.citation_label_lookup))
        span_labels.extend(index2label(span_label, dataset.span_label_lookup))
    
    return discourse_labels, citation_labels, span_labels


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--test_file', type=str, default="")
    argparser.add_argument('--related_work_file', type=str) # "20200705v1/acl/related_work.jsonl"
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=9999)
    argparser.add_argument('--output_file', type=str) # "tagged_related_works.jsonl"
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    args = argparser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    tokenizer.add_special_tokens(additional_special_tokens)

    params = vars(args)

    related_work_jsons = read_related_work_jsons(args.related_work_file)
    dev_set = JointRelatedWorkAnnotationDataset(args.test_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN)

    discourse_labels, citation_labels, span_labels = extract_label(dev_set)
    
    all_span_citation_mappings = annotate_related_work(discourse_labels, citation_labels, span_labels, dev_set, related_work_jsons, tokenizer)
    
    with open(args.output_file,"w") as f:
        for mapping in all_span_citation_mappings:
            json.dump(mapping,f)
            f.write("\n")