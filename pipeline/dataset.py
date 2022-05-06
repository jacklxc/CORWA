from glob import glob
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm import tqdm

from util import *

class JointPredictionDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, input_dict, tokenizer, MAX_SENT_LEN=512):
        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"pad": 0,
                                      "Intro": 1,
                                      "Single_summ": 2,
                                      "Multi_summ": 3,
                                      "Narrative_cite": 4,
                                      "Reflection": 5,
                                      "Transition": 1,
                                      "Other": 6
                                      }
        self.discourse_label_lookup = {0: "pad",
                                       1: "Transition",
                                       2: "Single_summ",
                                       3: "Multi_summ",
                                       4: "Narrative_cite",
                                       5: "Reflection",
                                       6: "Other"
                                       }

        self.span_label_types = {"pad": 0, "O": 1, "B_span": 2, "I_span": 3}
        self.span_label_lookup = {v: k for k, v in
                                  self.span_label_types.items()}

        self.citation_label_types = {"pad": 0, "O": 1, "B_Dominant": 2,
                                     "I_Dominant": 3, "B_Reference": 4,
                                     "I_Reference": 5}
        self.citation_label_lookup = {v: k for k, v in
                                      self.citation_label_types.items()}

        self.samples = []
        for paragraph_id, paragraph in input_dict.items():
            split_paragraphs, _, _ = read_paragraphs_split(
                [paragraph], tokenizer, self.max_sent_len)

            paragraph_ids = [paragraph_id + "_" + str(i) for i in range(len(split_paragraphs))]

            for pid, paragraph in zip(paragraph_ids, split_paragraphs):
                if paragraph:
                    N_tokens = len(
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"])
                    if N_tokens <= self.max_sent_len:
                        self.samples.append({
                            'id': pid,
                            'paragraph': paragraph.replace("[BOS]",
                                                           tokenizer.sep_token),
                            'N_tokens': N_tokens
                        })
                    else:
                        print("Unable to process paragraph ",pid)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    