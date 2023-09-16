import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from typing import List

import math
import numpy as np

from util import read_passages, clean_words, test_f1, to_BIO, from_BIO

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)
    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)
        
        return self.timedistributedlayer(x)
    
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        self.dense = TimeDistributedDense(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = TimeDistributedDense(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    
class WordAttention(nn.Module):
    """
    x: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
    token_mask: (batch_size, N_sep, N_token)
    out: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    mask: (BATCH_SIZE, N_sentence)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, dropout = 0.1):
        super(WordAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributedDense(PROJ_SIZE, 1)
        
    def forward(self, x, token_mask):
        proj_input = self.att_proj(self.dropout(x.view(-1, x.size(-1))))
        proj_input = self.dropout(self.activation(proj_input))
        raw_att_scores = self.att_scorer(proj_input).squeeze(-1).view(x.size(0),x.size(1),x.size(2)) # (Batch_size, N_sentence, N_token)
        att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
        batch_att_scores = att_scores.view(-1, att_scores.size(-1)) # (Batch_size * N_sentence, N_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1) 
        # (Batch_size * N_sentence, INPUT_SIZE)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        mask = token_mask[:,:,0]
        return out, mask
    
class JointParagraphTagger(nn.Module):
    def __init__(self, bert_path, tokenizer_len, dropout = 0.1, citation_label_size=6, span_label_size = 4, discourse_label_size = 7):
        super(JointParagraphTagger, self).__init__()
        self.citation_label_size = citation_label_size
        self.span_label_size = span_label_size
        self.discourse_label_size = discourse_label_size
        if "led-" in bert_path or "led_" in bert_path:
            led = AutoModelForSeq2SeqLM.from_pretrained(bert_path)
            led.resize_token_embeddings(tokenizer_len)
            self.bert = led.get_encoder()
            self.bert_dim = self.bert.config.d_model # bert_dim
        else:
            self.bert = AutoModel.from_pretrained(bert_path)
            self.bert.resize_token_embeddings(tokenizer_len)
            self.bert_dim = self.bert.config.hidden_size # bert_dim
        self.discourse_criterion = nn.CrossEntropyLoss(ignore_index = 0) #self.discourse_label_size)
        self.citation_criterion = nn.CrossEntropyLoss(ignore_index = 0) # self.citation_label_size)
        self.span_criterion = nn.CrossEntropyLoss(ignore_index = 0) #self.span_label_size)
        self.dropout = dropout
        self.word_attention = WordAttention(self.bert_dim, self.bert_dim, dropout=dropout)
        self.discourse_linear = ClassificationHead(self.bert_dim, self.discourse_label_size, hidden_dropout_prob = dropout)
        self.citation_linear = ClassificationHead(self.bert_dim, self.citation_label_size, hidden_dropout_prob = dropout)
        self.span_linear = ClassificationHead(self.bert_dim, self.span_label_size, hidden_dropout_prob = dropout)
        self.extra_modules = [
            self.word_attention,
            self.discourse_linear,
            self.citation_linear,
            self.span_linear,
            self.discourse_criterion,
            self.citation_criterion,
            self.span_criterion
        ]
    
    def forward(self, encoded_dict, transformation_indices, N_tokens, discourse_label = None, citation_label=None, span_label=None):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        #print(batch_indices.shape, indices_by_batch.shape, mask.shape)
        bert_out = self.bert(**encoded_dict)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        #print(bert_out.shape)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        #print(bert_tokens.shape)
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask)
        #sentence_reps = bert_tokens[:,:,0,:]
        #sentence_mask = mask[:,:,0]
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        
        discourse_out = self.discourse_linear(sentence_reps) # (Batch_size, N_sep, discourse_label_size)
        citation_out = self.citation_linear(bert_out)        
        span_out = self.span_linear(bert_out)
        
        if discourse_label is not None:
            discourse_loss = self.discourse_criterion(discourse_out.view(-1, self.discourse_label_size), discourse_label.view(-1))
            citation_loss = self.citation_criterion(citation_out.view(-1, self.citation_label_size), 
                                                      citation_label.view(-1))
            span_loss = self.span_criterion(span_out.view(-1, self.span_label_size), 
                                                      span_label.view(-1))
        else:
            discourse_loss = None
            citation_loss = None
            span_loss = None
            #loss = None
            
            
        discourse_pred = torch.argmax(discourse_out.cpu(), dim=-1) # (Batch_size, N_sep)
        discourse_out = [discourse_pred_paragraph[mask].detach().numpy().tolist() for discourse_pred_paragraph, mask in zip(discourse_pred, sentence_mask.bool().cpu())]
        citation_pred = torch.argmax(citation_out.cpu(), dim=-1) # (Batch_size, N_sep)
        citation_out = [citation_pred_paragraph[:n_token].detach().numpy().tolist() for citation_pred_paragraph, n_token in zip(citation_pred, N_tokens)]
        span_pred = torch.argmax(span_out.cpu(), dim=-1) # (Batch_size, N_sep)
        span_out = [span_pred_paragraph[:n_token].detach().numpy().tolist() for span_pred_paragraph, n_token in zip(span_pred, N_tokens)] 
        return discourse_out, citation_out, span_out, discourse_loss, citation_loss, span_loss
    
