"""Extracting frequent patterns given labels"""
import argparse
import json
import logging
import os
import pickle
from glob import glob
from typing import List

import pandas as pd

from config import classification_tasks
from pygapbide import Gapbide


def select_line_citation(line_citations:List) -> str:
    if len(line_citations) == 0:
        return ""
    if len(set(line_citations)) == 1:
        return line_citations[0]

    for x in ["Use", "Produce", "Extent", "Compare"]:
        if x in line_citations:
            return x
    return line_citations[0]


def get_gapbide_patterns(processed_docs, min_sup=10, min_gap=0, max_gap=0):
    g = Gapbide(processed_docs, min_sup, min_gap, max_gap)
    patterns = g.run()
    return patterns, g


def log_top_patterns(patterns, count=20):
    print([x for x in sorted(patterns, reverse=True, key=lambda x: x[1]) if
           len(x[0]) > 1][:count])


DISCOURSE_LABEL_TYPE = {"Intro": 0,
                        "Single_summ": 1,
                        "Multi_summ": 2,
                        "Narrative_cite": 3,
                        "Reflection": 4,
                        "Transition": 5,
                        "Other": 6
                        }
CITATION_TYPES = {"Dominant": 0, "Reference": 1}
SPAN_ANNOTATION_TYPES = {"B_span".lower(): 0, "E_span".lower(): 1}


def get_base_filename(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_annotated_file(file, folder, ext):
    return os.path.join(folder,
                        get_base_filename(file) + ".{}".format(ext))


class Annotation(object):

    def __init__(self, text_file=None):
        """
        :param text_file: TEXT file PATH
        """
        self.text_file = text_file
        self.ann_file = text_file.replace(".txt", ".ann")
        self.text = self._read_text()
        if not os.path.exists(self.text_file):
            logging.error("{} doesn't exist".format(self.text_file))
        if not os.path.exists(self.ann_file):
            logging.error("{} doesn't exist".format(self.ann_file))

    def _read_text(self):
        with open(self.text_file) as f:
            return f.read()

    def get_discourse_labels(self):
        annotations = []
        with open(self.ann_file) as f:
            for line in f:
                ID, content, text = line.strip().split("\t")
                if text != "[BOS]":
                    continue
                content = content.split()
                label = content[0]
                if label in DISCOURSE_LABEL_TYPE:
                    start = int(content[1])
                    end = int(content[2])
                    annotations.append((start, end, label, text))
        annotations = sorted(annotations, key=lambda x: x[0])
        return annotations

    def get_citation_types(self):
        """Get citation types"""
        annotations = []
        with open(self.ann_file) as f:
            for line in f:
                ID, content, text = line.strip().split("\t")
                content = content.split()
                label = content[0]

                if label in CITATION_TYPES:
                    start = int(content[1])
                    end = int(content[2])
                    annotations.append((start, end, label, text))
        annotations = sorted(annotations, key=lambda x: x[0])
        return annotations

    @staticmethod
    def get_citation_function_data(file):
        if os.path.exists(file):
            with open(file, "r") as reader:
                return pd.DataFrame(json.loads(reader.read()))
        else:
            return None

    def get_merged_citation_functions(self):
        pass

    def get_span_annotations(self) -> List:
        """
        Get span annotations -->
        Returns:list of (start, end, text) pairs
        """
        annotations = []
        B_SPAN = "b_span"
        E_SPAN = "e_span"

        stack = []
        start_stack = []

        with open(self.ann_file) as f:
            for line in f:
                ID, content, text = line.strip().split("\t")
                content = content.split()
                label = content[0].lower()

                if label in SPAN_ANNOTATION_TYPES:
                    if label == B_SPAN:
                        start = int(content[1])
                        stack.append(label)
                        start_stack.append(start)

                    elif label == E_SPAN:
                        if stack.__len__() > 0:
                            end = int(content[2])
                            stack.pop()
                            start = start_stack.pop()
                            annotations.append(
                                (start, end, self.text[start:end]))

        annotations = sorted(annotations, key=lambda x: x[0])
        return annotations

    @staticmethod
    def get_span_token(ann, tokenizer):
        """
        Args:
            ann: list of text
            tokenizer: tokenizer
        Returns: List of tokens
        """
        all_tokens = []
        for text in ann:
            all_tokens.extend(tokenizer.tokenize(text))
        return all_tokens


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Predict citation functions!")

    argparser.add_argument('--discourse', dest='discourse', action='store_true')
    argparser.set_defaults(discourse=False)
    argparser.add_argument('--discourse_folder', type=str)

    argparser.add_argument('--citation_function', dest='citation_function',
                           action='store_true')
    argparser.set_defaults(citation_function=False)
    argparser.add_argument('--citation_function_folder', type=str)

    argparser.add_argument('--scicite', dest='scicite', action='store_true')
    argparser.set_defaults(scicite=False)
    argparser.add_argument('--scicite_folder', type=str)

    argparser.add_argument('--scires', dest='scires', action='store_true')
    argparser.set_defaults(scires=False)
    argparser.add_argument('--scires_folder', type=str)

    argparser.add_argument('--save_path', type=str)

    logging.getLogger("transformers.tokenization_utils_base").setLevel(
        logging.ERROR)

    args = argparser.parse_args()

    params = vars(args)

    for k, v in params.items():
        print(k, v)
    discourse_flag, scires_flag, scicite_flag, \
    citation_function_flag = args.discourse, args.scires, args.scicite, args.citation_function


    def update_merged_data(merged_data: pd.DataFrame, data):
        if merged_data is None:
            return data
        else:
            if data:
                return merged_data.merge(
                    data,
                    on=["paper_id", "citation_tag", "start", "end"],
                    suffixes=("", "_y")
                )
            else:
                return merged_data


    processed_patterns, processed_sentences, citations = [], [], []

    print("Discourse folder source: {}".format(args.discourse_folder))
    for file in glob(os.path.join(args.discourse_folder, "*.txt")):
        annotatation = Annotation(file)

        discourse_data, merged_data, scires_data, scicite_data, \
        citation_function_data = None, None, None, None, None
        if discourse_flag:
            discourse_data = annotatation.get_discourse_labels()

        if scires_flag:
            scires_data = annotatation.get_citation_function_data(
                get_annotated_file(file, args.scires_folder,
                                   classification_tasks[2]))
            merged_data = update_merged_data(merged_data, scires_data)

        if scicite_flag:
            scicite_data = annotatation.get_citation_function_data(
                get_annotated_file(file, args.scicite_folder,
                                   classification_tasks[0]))
            merged_data = update_merged_data(merged_data, scicite_data)

        if citation_function_flag:
            citation_function_data = annotatation.get_citation_function_data(
                get_annotated_file(file, args.citation_function_folder,
                                   classification_tasks[1]))
            merged_data = update_merged_data(merged_data,
                                             citation_function_data)

        paragraphs = annotatation.text.split("\n\n")

        if merged_data is not None:
            merged_data.sort_values(by="start", inplace=True)
            merged_data = merged_data.to_dict("records")

        i, j = 0, 0
        start, end = 0, 0
        for par in paragraphs:
            paragraph_patterns, paragraph_sentences = [], []

            for line in par.splitlines():
                # all labels, only discourse, only citations(scires, scicite, citation function)
                line_patterns, discourse_patterns_line, line_citations = [], [], []

                end = start + len(line)
                if discourse_data and i < len(discourse_data):
                    d = discourse_data[i]
                    if d[0] >= start and d[1] <= end:
                        line_patterns.append(d[2])
                        discourse_patterns_line.append(d[2])
                        i += 1
                        if i == len(discourse_data):
                            break

                if merged_data is not None and j < len(merged_data) and \
                        merged_data[j]["start"] <= end:
                    while start <= merged_data[j]["start"] <= end:
                        d = merged_data[j]
                        for task in classification_tasks:
                            label_key = "{}_label".format(task)
                            if label_key.format(task) in d:
                                line_patterns.append(d[label_key])
                                line_citations.append(d[label_key])
                        j += 1
                        if j == len(merged_data):
                            break
                if len(line_patterns) > 0:
                    # paragraph_patterns.append("__".join(line_patterns))
                    pt = []
                    if len(discourse_patterns_line)>0:
                        pt.append(discourse_patterns_line[0])
                    ct = select_line_citation(line_citations)
                    if ct:
                        pt.append(ct)

                    paragraph_patterns.append("__".join(pt))
                    # for _ in line_patterns:
                    paragraph_sentences.append(line)
                    # paragraph_sentences.append(line)
                start += len(line) + 1
                citations.append(line_citations)
            if len(paragraph_patterns) > 0:
                processed_patterns.append(paragraph_patterns)
                processed_sentences.append(paragraph_sentences)

            # paragraph end has extra newline
            start += 1
    print(len(processed_patterns))
    patterns, g = get_gapbide_patterns(processed_docs=processed_patterns,
                                       min_gap=0, max_gap=0)

    log_top_patterns(patterns, count=200)

    if args.save_path:
        with open(args.save_path, 'wb') as fp:
            pickle.dump(
                {"sentence": processed_sentences, "g": g, "patterns": patterns},
                fp, protocol=pickle.HIGHEST_PROTOCOL)
