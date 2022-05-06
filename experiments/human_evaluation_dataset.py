#!/usr/bin/env python
# coding: utf-8


import json
import os
import sys
from copy import deepcopy

from transformers import AutoTokenizer

device = "cuda"

max_input_length = 16384
max_output_length = 1024

# In[13]:


path = sys.argv[1]
print("Model path.. ", path)

tokenizer = AutoTokenizer.from_pretrained(path)
special_tokens = ['<doc>', '</doc>', '[BOS]', '[Dominant]', '[Reference]',
                  '[B_Dominant]', '[E_Dominant]', '[B_Reference]',
                  '[E_Reference]']
additional_special_tokens = {'additional_special_tokens': special_tokens}
tokenizer.add_special_tokens(additional_special_tokens)

sample_path = sys.argv[2]
print("sample data path.. ", sample_path)

generation_type = sys.argv[3]


def get_references_list(src):
    """Get citations given source content"""
    all_citations, all_titles, all_abstracts = [], [], []
    for cite_data in src.split("[B_Reference]")[1:]:

        all_citations.append(cite_data.split("</s>")[0].strip())

        title_abstract = cite_data.split("</s>")[1]

        ref_data = title_abstract[:title_abstract.index("[E_Reference]")]
        try:
            all_titles.append(
                ref_data.split("|")[0].strip() if len(
                    ref_data.split("|")) > 0 else ""
            )
            all_abstracts.append(
                ref_data.split("|")[1].strip() if len(
                    ref_data.split("|")) > 1 else ""
            )
        except:
            print(cite_data)

    for cite_data in src.split("[B_Dominant]")[1:]:

        all_citations.append(cite_data.split("</s>")[0].strip())

        title_abstract = cite_data.split("</s>")[1]

        ref_data = title_abstract[:title_abstract.index("[E_Dominant]")]
        try:
            all_titles.append(
                ref_data.split("|")[0].strip() if len(
                    ref_data.split("|")) > 0 else ""
            )
            all_abstracts.append(
                ref_data.split("|")[1].strip() if len(
                    ref_data.split("|")) > 1 else ""
            )
        except:
            print(cite_data)

    return all_citations, all_titles, all_abstracts


def map_references(cit_marks, titles, abstracts):
    references = []
    for cit_mark, title, abstract in zip(cit_marks, titles, abstracts):
        references.append({"citation_mark": cit_mark,
                           "title": title,
                           "abstract": abstract})
    return references

def get_reference_titles(references):
    titles = ""

    for d in references:
        if "title" in d:
            titles += d["title"].strip()
    return titles


def get_total_citation_mark_length(references):
    return " ".join(
        [x["citation_mark"] for x in references]
    ).__len__()


def get_current_paper_abstract_intro(source, split_token="</s>"):
    return source.split(split_token)[0]


def get_related_work(source, split_token="</s>"):
    """get the related work section given the overall source"""

    context = source.split(split_token)[1]

    dominant_index, reference_index = len(context), len(context)

    if "[B_Dominant]" in context:
        dominant_index = context.index("[B_Dominant]")

    if "[B_Reference]" in context:
        reference_index = context.index("[B_Reference]")

    return context[:min(dominant_index, reference_index)].strip()


def get_current_paper_context(source, split_token="</s>"):
    return source.split(split_token)[0]


def new_uid(uid, gen_type, num):
    return "{}_{}_{}".format(uid, gen_type, num)


def get_citation_label(source, mask_token):
    """get the citation label(or return 
        sentence otherwise) from source"""

    if "[Dominant]" in source:
        return "Dominant"

    if "[Reference]" in source:
        return "Reference"

    if mask_token in source:
        return "Sentence"


def validate_generation_type(gen_type):
    if gen_type not in ["sentence", "span"]:
        raise ValueError(
            "third argument must be one of these {}".format(
                ["sentence", "span"])
        )


validate_generation_type(generation_type)

with open(os.path.join(sample_path, "sample_output.json"), 'r') as f:
    content = json.loads(f.read())

human_eval_content = []
citation_counter = dict()
PER_CITATION_LIMIT = 10000
replacement_pattern = "$$$$$$$$$$$$$${}$$$$$$$$$$$$$$"

for i, data in enumerate(content):
    cit_marks, titles, abstracts = get_references_list(data["source"])

    assert len(cit_marks) == len(titles), "MISMATCH {}\n{}\n{}\n{}".format(
        data, cit_marks, titles, abstracts)
    assert len(titles) == len(
        abstracts), "MISMATCH {}\n{}\n{}\n{}\n\n\n".format(
        data, cit_marks, titles, abstracts)

    if data["target"].split("\n").__len__() > 1 \
            or data["generated"].split("\n").__len__() > 1:
        continue

    sample = dict()
    references = map_references(cit_marks, titles, abstracts)

    if get_reference_titles(references).__len__() < 1:
        continue

    citation_mark_length = get_total_citation_mark_length(sample)

    if len(data["target"]) < citation_mark_length + 10:
        continue

    sample["introduction"] = get_current_paper_abstract_intro(
        source=data["source"]
    )

    related_work = get_related_work(data["source"])

    # sample.update({"fluency": 0,
    #                "relevance": 0,
    #                "coherence": 0,
    #                "overall": 0})

    citation_label = get_citation_label(
        source=data["source"],
        mask_token=tokenizer.mask_token
    )
    sample["citation_label"] = citation_label
    citation_counter[citation_label] = citation_counter.get(citation_label,
                                                            0) + 1

    if citation_counter[citation_label] > PER_CITATION_LIMIT:
        continue

    # Actual target
    actual_related_work = related_work
    for mask in ["[Dominant]", "[Reference]", tokenizer.mask_token]:
        actual_related_work = actual_related_work.replace(
            mask, replacement_pattern.format(data["target"])
        )

    sample["id"] = new_uid(data["part_id"], generation_type, 0)
    sample["related_work"] = actual_related_work
    sample["references"] = references
    human_eval_content.append(sample)

    sample = deepcopy(sample)

    # Generated data
    generated_related_work = related_work
    for mask in ["[Dominant]", "[Reference]", tokenizer.mask_token]:
        generated_related_work = generated_related_work.replace(
            mask, replacement_pattern.format(data["generated"])
        )

    sample["id"] = new_uid(data["part_id"], generation_type, 1)
    sample["related_work"] = generated_related_work
    human_eval_content.append(sample)

human_eval_content.__len__()

with open(os.path.join(sample_path, "human_evaluation.json"), "w") as f:
    json.dump(human_eval_content, f)
