import argparse

import torch
import jsonlines
import os
import pickle
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, \
    get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import flatten, crossvalid
from paragraph_model import JointParagraphTagger
# from paragraph_model import JointParagraphCRFTagger as JointParagraphTagger
from dataset import JointRelatedWorkAnnotationDataset

import logging


def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def batch_token_label(labels, padding_idx):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        label_indices = [int(evid) for evid in label]
        label_matrix[i, :len(label_indices)] = torch.tensor(label_indices)
        label_list.append(label_indices)
    return label_matrix.long(), label_list


# def batch_discourse_label(labels, padding_idx):
#    max_paragraph_len = max([len(label) for label in labels])
#    label_matrix = torch.ones(len(labels), max_paragraph_len) * padding_idx
#    label_list = []
#    for i, label in enumerate(labels):
#        for j, evid in enumerate(label):
#            label_matrix[i,j] = int(evid)
#        label_list.append([int(evid) for evid in label])
#    return label_matrix.long(), label_list

def index2label(all_indices, mapping):
    all_labels = []
    for indices in all_indices:
        all_labels.append([mapping.get(index, "pad") for index in indices])
    return all_labels


def predict(model, dataset):
    model.eval()
    discourse_predictions = []
    citation_predictions = []
    span_predictions = []

    with torch.no_grad():
        for batch in tqdm(
                DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch,
                                  has_local_attention="led-" in args.repfile or "longformer" in args.repfile or "led_" in args.repfile)
            transformation_indices = token_idx_by_sentence(
                encoded_dict["input_ids"],
                tokenizer.sep_token_id)
            encoded_dict = {key: tensor.to(device) for key, tensor in
                            encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in
                                      transformation_indices]
            discourse_out, citation_out, span_out, _, _, _ = model(encoded_dict,
                                                                   transformation_indices,
                                                                   batch[
                                                                       "N_tokens"])
            discourse_predictions.extend(
                index2label(discourse_out, dataset.discourse_label_lookup))
            citation_predictions.extend(
                index2label(citation_out, dataset.citation_label_lookup))
            span_predictions.extend(
                index2label(span_out, dataset.span_label_lookup))
    return discourse_predictions, citation_predictions, span_predictions


def evaluation_metric(discourse_labels, discourse_predictions, mapping):
    positive_labels = []
    for label in mapping.keys():
        if label not in {"Other", "O", "pad"}:
            positive_labels.append(label)

    flatten_labels = flatten(discourse_labels)
    flatten_predictions = flatten(discourse_predictions)
    if len(flatten_labels) != len(flatten_predictions):
        min_len = min([len(flatten_labels), len(flatten_predictions)])
        flatten_labels = flatten_labels[:min_len]
        flatten_predictions = flatten_predictions[:min_len]
    discourse_f1 = f1_score(flatten_labels, flatten_predictions,
                            average='micro', labels=positive_labels)
    discourse_precision = precision_score(flatten_labels, flatten_predictions,
                                          average='micro',
                                          labels=positive_labels)
    discourse_recall = recall_score(flatten_labels, flatten_predictions,
                                    average='micro', labels=positive_labels)
    return (discourse_f1, discourse_recall, discourse_precision)


def evaluation(model, dataset):
    model.eval()
    discourse_predictions = []
    discourse_labels = []
    citation_predictions = []
    citation_labels = []
    span_predictions = []
    span_labels = []

    with torch.no_grad():
        for batch in tqdm(
                DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            try:
                encoded_dict = encode(tokenizer, batch,
                                      has_local_attention="led-" in args.repfile or "longformer" in args.repfile or "led_" in args.repfile)
                transformation_indices = token_idx_by_sentence(
                    encoded_dict["input_ids"],
                    tokenizer.sep_token_id)
                encoded_dict = {key: tensor.to(device) for key, tensor in
                                encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in
                                          transformation_indices]

                padded_discourse_label, discourse_label = batch_token_label(
                    batch["discourse_label"], 0)
                padded_citation_label, citation_label = batch_token_label(
                    batch["citation_label"], 0)
                padded_span_label, span_label = batch_token_label(
                    batch["span_label"], 0)  # len(dev_set.span_label_types))
                discourse_out, citation_out, span_out, _, _, _ = \
                    model(encoded_dict, transformation_indices,
                          batch["N_tokens"],
                          discourse_label=padded_discourse_label.to(device),
                          citation_label=padded_citation_label.to(device),
                          span_label=padded_span_label.to(device),
                          )

                discourse_predictions.extend(
                    index2label(discourse_out, dataset.discourse_label_lookup))
                discourse_labels.extend(index2label(discourse_label,
                                                    dataset.discourse_label_lookup))
                citation_predictions.extend(
                    index2label(citation_out, dataset.citation_label_lookup))
                citation_labels.extend(
                    index2label(citation_label, dataset.citation_label_lookup))
                span_predictions.extend(
                    index2label(span_out, dataset.span_label_lookup))
                span_labels.extend(
                    index2label(span_label, dataset.span_label_lookup))
            except:
                pass

    return evaluation_metric(discourse_labels, discourse_predictions,
                             dataset.discourse_label_types), \
           evaluation_metric(citation_labels, citation_predictions,
                             dataset.citation_label_types), \
           evaluation_metric(span_labels, span_predictions,
                             dataset.span_label_types)


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
    argparser = argparse.ArgumentParser(
        description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str,
                           default="allenai/scibert_scivocab_uncased",
                           help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="")
    argparser.add_argument('--distant_file', type=str, default="")
    argparser.add_argument('--k_fold', type=int, default=5)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--pre_trained_model_resize',
                           dest='pre_trained_model_resize',
                           action='store_true')
    argparser.set_defaults(pre_trained_model_resize=False)
    # argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="")
    argparser.add_argument('--bert_lr', type=float, default=1e-5,
                           help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6,
                           help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0,
                           help="embedding_dropout rate")
    # argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15,
                           help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str,
                           default="joint_tagger.model")
    argparser.add_argument('--log_file', type=str,
                           default="joint_tagger_performances.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int,
                           default=1)  # roberta-large: 2; bert: 8
    argparser.add_argument('--discourse_coef', type=float, default=1)  # 1
    argparser.add_argument('--citation_coef', type=float, default=1)  # 1.5
    argparser.add_argument('--span_coef', type=float, default=1)  # 2.5
    logging.getLogger("transformers.tokenization_utils_base").setLevel(
        logging.ERROR)

    reset_random_seed(12345)

    args = argparser.parse_args()
    # device = "cpu" ###############################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    tokenizer.add_special_tokens(additional_special_tokens)

    if args.train_file:
        train = True
        # assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False
    if args.test_file:
        test = True
    else:
        test = False

    params = vars(args)

    for k, v in params.items():
        print(k, v)

    if train:
        all_train = JointRelatedWorkAnnotationDataset(args.train_file,
                                                      tokenizer,
                                                      MAX_SENT_LEN=args.MAX_SENT_LEN)
        if args.distant_file is not None:
            all_distant = JointRelatedWorkAnnotationDataset(args.distant_file,
                                                            tokenizer,
                                                            MAX_SENT_LEN=args.MAX_SENT_LEN,
                                                            dummy_discourse=False,
                                                            dummy_span=False,
                                                            dummy_citation=False)  #####
            all_train.merge(all_distant)
    test_set = JointRelatedWorkAnnotationDataset(args.test_file, tokenizer,
                                                MAX_SENT_LEN=args.MAX_SENT_LEN)

    for tr, dv in crossvalid(all_train, k_fold=args.k_fold):
        train_set, dev_set = deepcopy(all_train), deepcopy(all_train)
        train_set.update_samples(tr)
        dev_set.update_samples(dv)
        print("Train, dev data sizes : ({}, {})".format(len(train_set), len(dev_set)))


        if not (
                args.pre_trained_model is not None and args.pre_trained_model_resize):
            model = JointParagraphTagger(args.repfile, len(tokenizer),
                                         args.dropout)  # .to(device)

        if args.pre_trained_model is not None:
            if args.pre_trained_model_resize:
                model = JointParagraphTagger(args.repfile, len(tokenizer),
                                             args.dropout,
                                             bert_pretrained_model=args.pre_trained_model)  # .to(device)
            else:
                model.load_state_dict(torch.load(args.pre_trained_model))

        model = model.to(device)

        if train:
            settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
            for module in model.extra_modules:
                settings.append({'params': module.parameters(), 'lr': args.lr})
            optimizer = torch.optim.Adam(settings)
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)
            model.train()

            prev_performance = 0
            for epoch in range(args.epoch):
                tq = tqdm(
                    DataLoader(train_set, batch_size=args.batch_size, shuffle=True))
                for i, batch in enumerate(tq):
                    encoded_dict = encode(tokenizer, batch,
                                          has_local_attention="led-" in args.repfile or "longformer" in args.repfile or "led_" in args.repfile)
                    transformation_indices = token_idx_by_sentence(
                        encoded_dict["input_ids"],
                        tokenizer.sep_token_id)
                    encoded_dict = {key: tensor.to(device) for key, tensor in
                                    encoded_dict.items()}
                    transformation_indices = [tensor.to(device) for tensor in
                                              transformation_indices]
                    padded_discourse_label, discourse_label = batch_token_label(
                        batch["discourse_label"], 0)

                    padded_citation_label, citation_label = batch_token_label(
                        batch["citation_label"], 0)
                    padded_span_label, span_label = batch_token_label(
                        batch["span_label"], 0)  # len(dev_set.span_label_types))
                    discourse_out, citation_out, span_out, discourse_loss, citation_loss, span_loss = \
                        model(encoded_dict, transformation_indices,
                              batch["N_tokens"],
                              discourse_label=padded_discourse_label.to(device),
                              citation_label=padded_citation_label.to(device),
                              span_label=padded_span_label.to(device),
                              )
                    loss = discourse_loss * args.discourse_coef + citation_loss * args.citation_coef + span_loss * args.span_coef
                    loss.backward()

                    if i % args.update_step == args.update_step - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        tq.set_description(
                            f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
                scheduler.step()

                # Evaluation
                train_discourse_score, train_citation_score, train_span_score = evaluation(
                    model, train_set)
                print(
                    f'Epoch {epoch}, train discourse f1 p r: %.4f, %.4f, %.4f' % train_discourse_score)
                print(
                    f'Epoch {epoch}, train citation f1 p r: %.4f, %.4f, %.4f' % train_citation_score)
                print(
                    f'Epoch {epoch}, train span f1 p r: %.4f, %.4f, %.4f' % train_span_score)

                dev_discourse_score, dev_citation_score, dev_span_score = evaluation(
                    model, dev_set)
                print(
                    f'Epoch {epoch}, dev discourse f1 p r: %.4f, %.4f, %.4f' % dev_discourse_score)
                print(
                    f'Epoch {epoch}, dev citation f1 p r: %.4f, %.4f, %.4f' % dev_citation_score)
                print(
                    f'Epoch {epoch}, dev span f1 p r: %.4f, %.4f, %.4f' % dev_span_score)

                dev_perf = dev_discourse_score[0] * dev_citation_score[0] * \
                           dev_span_score[0]
                if dev_perf >= prev_performance:
                    torch.save(model.state_dict(), args.checkpoint)
                    best_state_dict = model.state_dict()
                    prev_performance = dev_perf
                    best_scores = (
                    dev_discourse_score, dev_citation_score, dev_span_score)
                    print("New model saved!")
                else:
                    print("Skip saving model.")

            # torch.save(model.state_dict(), args.checkpoint)
            params["discourse_f1"] = params.get("discourse_f1", 0) + best_scores[0][0] / args.k_fold
            params["discourse_precision"] = params.get("discourse_precision", 0) + best_scores[0][1] / args.k_fold
            params["discourse_recall"] = params.get("discourse_recall", 0) + best_scores[0][2] / args.k_fold

            params["citation_f1"] = params.get("citation_f1", 0) + best_scores[1][0] / args.k_fold
            params["citation_precision"] = params.get("citation_precision", 0) + best_scores[1][1] / args.k_fold
            params["citation_recall"] = params.get("citation_recall", 0) + best_scores[1][2] / args.k_fold

            params["span_f1"] = params.get("span_f1", 0) + best_scores[2][0] / args.k_fold
            params["span_precision"] = params.get("span_precision", 0) + best_scores[2][1] / args.k_fold
            params["span_recall"] = params.get("span_recall", 0) + best_scores[2][2] / args.k_fold

    with jsonlines.open(args.log_file, mode='a') as writer:
        writer.write(params)