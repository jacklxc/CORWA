import argparse

import json
from tqdm import tqdm
import numpy as np

import os
import sys

from util import *
from data_util import *
from joint_tagger import run_prediction

from transformers import AutoTokenizer
     
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--related_work_file', type=str) # "20200705v1/acl/related_work.jsonl"
    argparser.add_argument('--output_file', type=str) # "tagged_related_works.jsonl"
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "joint_tagger_train_scibert_final.model")
    argparser.add_argument('--batch_size', type=int, default=32) # roberta-large: 2; bert: 8
    args = argparser.parse_args()
        
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    related_work_jsons = read_related_work_jsons(args.related_work_file)
        
    paragraphs = {}
    for paper_id, related_work_dict in related_work_jsons.items():
        for pi, para in enumerate(related_work_dict["related_work"]):
            paragraph_id = paper_id + "_" + str(pi)
            paragraphs[paragraph_id] = " ".join(scientific_sent_tokenize(para["text"]))

    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    tokenizer.add_special_tokens(additional_special_tokens)

    discourse_predictions, citation_predictions, span_predictions, dataset = run_prediction(paragraphs, tokenizer, args)

    all_span_citation_mappings = annotate_related_work(discourse_predictions, citation_predictions, span_predictions, dataset, related_work_jsons, tokenizer)
    
    with open(args.output_file,"w") as f:
        for mapping in all_span_citation_mappings:
            json.dump(mapping,f)
            f.write("\n")