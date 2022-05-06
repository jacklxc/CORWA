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
from util import flatten, fix_BIO, removeAccents, citation_prediction_to_annotation_paragraph, span_prediction_to_annotation_paragraph, discourse_prediction_to_annotation_paragraph, paragraph2doc_annotation, merge_annotations_by_doc, write_brat, post_process_spans, realign_samples
from paragraph_model import JointParagraphTagger
# from paragraph_model import JointParagraphCRFTagger as JointParagraphTagger
from dataset import JointRelatedWorkAnnotationDataset

import logging

import pickle

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def index2label(all_indices, mapping):
    all_labels = []
    for indices in all_indices:
        all_labels.append([mapping.get(index,"pad") for index in indices])
    return all_labels

def predict(model, dataset):
    model.eval()
    paragraph_ids = []
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
                paragraph_ids.extend(batch["id"])
                discourse_predictions.extend(index2label(discourse_out, dataset.discourse_label_lookup))
                citation_predictions.extend(index2label(citation_out, dataset.citation_label_lookup))
                span_predictions.extend(index2label(span_out, dataset.span_label_lookup))
            except:
                pass
    return paragraph_ids, discourse_predictions, citation_predictions, span_predictions


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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--test_file', type=str, default="")
    argparser.add_argument('--output_file', type=str, default="")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "joint_tagger.model")
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    tokenizer.add_special_tokens(additional_special_tokens)

    params = vars(args)

    for k,v in params.items():
        print(k,v)
    
    dev_set = JointRelatedWorkAnnotationDataset(args.test_file, tokenizer, MAX_SENT_LEN = args.MAX_SENT_LEN, train=False)

    model = JointParagraphTagger(args.repfile, len(tokenizer),
                                      args.dropout)#.to(device)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
        print("Model loaded!")
    
    model = model.to(device)
    paragraph_ids, discourse_predictions, citation_predictions, span_predictions = predict(model, dev_set)
    citation_predictions = fix_BIO(citation_predictions)
    span_predictions = fix_BIO(span_predictions)
    span_predictions = post_process_spans(span_predictions, citation_predictions) # Still a little bit buggy
    
    with open("discourse_predictions.pkl","wb") as f:
        pickle.dump(discourse_predictions, f)
    with open("citation_predictions.pkl","wb") as f:
        pickle.dump(citation_predictions, f)
    with open("span_predictions.pkl","wb") as f:
        pickle.dump(span_predictions, f)
    with open("dataset.pkl","wb") as f:
        pickle.dump(dev_set, f)
    
    sample_indices = realign_samples(dev_set, paragraph_ids)
    all_citation_annotations = citation_prediction_to_annotation_paragraph(dev_set, citation_predictions, tokenizer, sample_indices = sample_indices)
    paper_ids, all_texts, citation_annotation_by_doc = paragraph2doc_annotation(dev_set, all_citation_annotations, tokenizer, sample_indices = sample_indices)
    all_span_annotations = span_prediction_to_annotation_paragraph(dev_set, span_predictions, tokenizer, sample_indices = sample_indices)
    paper_ids, all_texts, span_annotation_by_doc = paragraph2doc_annotation(dev_set, all_span_annotations, tokenizer, sample_indices = sample_indices)
    all_discourse_annotations = discourse_prediction_to_annotation_paragraph(dev_set, discourse_predictions, tokenizer, sample_indices = sample_indices)
    paper_ids, all_texts, discourse_annotation_by_doc = paragraph2doc_annotation(dev_set, all_discourse_annotations, tokenizer, sample_indices = sample_indices)
    merged_annotations = merge_annotations_by_doc(discourse_annotation_by_doc, citation_annotation_by_doc, span_annotation_by_doc)
    write_brat(args.output_file, paper_ids, all_texts, merged_annotations)