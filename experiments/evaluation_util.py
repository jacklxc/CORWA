import json

import pandas as pd


def check_annotated(ann_list):
    for ann in ann_list:
        if ann["fluency"] == 0:
            return False
        if ann["relevance"] == 0:
            return False
        if ann["coherence"] == 0:
            return False
        if ann["overall"] == 0:
            return False
    return True


class HumanEvaluator(object):

    def __init__(self, file, all_order_mapping, evaluator_name=""):
        self.file = file
        self.evaluator_name = evaluator_name
        self.all_order_mapping = all_order_mapping
        self.sentence_gen_data, self.span_gen_data, self.span_gold_data = [], [], []
        self.annotation = None
        self.load_file()
        self.parse_annotation()

    def load_file(self):
        with open(self.file, 'r') as handle:
            self.annotation = json.loads(handle.read())
        return self.annotation

    def parse_annotation(self):
        for d in self.annotation:
            seq = self.all_order_mapping[d["id"]]
            related_work_tagged_list = d["related_works"]
            if not check_annotated(related_work_tagged_list):
                continue

            # span gold
            span_gold = related_work_tagged_list[seq.index(1)]
            span_gold["id"] = d["id"]
            self.span_gold_data.append(span_gold)

            # span generated
            span_gen = related_work_tagged_list[seq.index(2)]
            span_gen["id"] = d["id"]
            self.span_gen_data.append(span_gen)

            # sentence generated
            sent_gen = related_work_tagged_list[seq.index(3)]
            sent_gen["id"] = d["id"]
            self.sentence_gen_data.append(sent_gen)

    def get_span_gold_dataframe(self):
        return pd.DataFrame(self.span_gold_data)

    def get_span_generation_dataframe(self):
        return pd.DataFrame(self.span_gen_data)

    def get_sentence_generation_dataframe(self):
        return pd.DataFrame(self.sentence_gen_data)
