from glob import glob
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm import tqdm

from config import CITATION_TYPES
from util import *


class CORWADatasetTextFile(Dataset):

    def __init__(self, folder, tokenizer, citation_types=None, context_window=2,
                 citation_replacement=None, MAX_SENT_LEN=512) -> None:
        self.max_sent_len = MAX_SENT_LEN
        if citation_types is None:
            self.citation_types = CITATION_TYPES
        else:
            self.citation_types = citation_types

        files = glob(os.path.join(folder, "*.txt"))

        self.samples = []

        for file in files:
            ann_file = file.replace(".txt", ".ann")

            with open(file, "r") as textread:
                file_text = textread.read()
            if not os.path.exists(ann_file):
                raise FileNotFoundError("file: {}".format(ann_file))

            with open(ann_file, "r") as annread:
                ann_text = annread.read()

            citations = []
            for line in ann_text.splitlines():
                tag_no, tags, text = line.split("\t")
                tag, start, end = tags.split()
                if tag in self.citation_types:
                    start, end = int(start), int(end)
                    citations.append((start, end, tag, text))

            # sort based on starting index
            citations = sorted(citations, key=lambda x: x[0])

            position, i = 0, 0
            paragraphs = file_text.split("\n\n")
            next_par_position = paragraphs[0].__len__() + 2

            for start, end, tag, text in citations:

                if start >= next_par_position:
                    while start >= next_par_position:
                        i += 1
                        position = next_par_position
                        next_par_position += paragraphs[i].__len__() + 2

                paragraph = file_text[position: next_par_position]
                par_start, par_end = start - position, end - position
                context_text = paragraph[
                               :par_start] + "_CITE_" + paragraph[
                                                        par_end:]

                context_sents = context_text.splitlines()
                for idx, ln in enumerate(context_sents):
                    if "_CITE_" in ln:
                        context_window_text = context_sents[
                                              max(idx - context_window, 0):min(
                                                  idx + context_window + 1, len(
                                                      context_sents))]
                        context = "".join(context_window_text).replace(
                            "[BOS]", "").replace("\n", " ").strip()
                        if citation_replacement is None:
                            context = context.replace("_CITE_", text)
                        else:
                            context = context.replace("_CITE_",
                                                      citation_replacement)

                        offset_mapping = \
                            tokenizer(context,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]

                        if len(offset_mapping) > self.max_sent_len:
                            context = context[
                                      :offset_mapping[self.max_sent_len - 2][
                                          -1]]

                        self.samples.append({
                            "paper_id": os.path.splitext(
                                os.path.basename(file)
                            )[0],
                            "citation_tag": tag,
                            "start": start,
                            "end": end,
                            "citation_text": text,
                            "sentence": context
                        })
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class JointRelatedWorkAnnotationDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=512,
                 sentence_overlap=0, dummy_discourse=False, dummy_span=False,
                 dummy_citation=False):
        self.max_sent_len = MAX_SENT_LEN
        # If a paragraph is truncated, how many sentences to overlap, so that the context information is preserved?
        self.sentence_overlap = sentence_overlap
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

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.samples = []
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))
            if train:
                if "requirements" in text_file:
                    continue
                try:
                    annotation_file = text_file.replace(".txt", ".ann")
                    all_annotations = read_annotations(annotation_file, offsets)

                    for paragraph_id, paragraph, paragraph_annotation in zip(
                            paragraph_ids, paragraphs, all_annotations):
                        for annotation in paragraph_annotation:
                            assert paragraph[annotation[0]:annotation[1]] == \
                                   annotation[-1]
                        # paragraph = paragraph.lower().replace("[bos]","[BOS]") ######
                        # N_tokens = len(tokenizer.tokenize(paragraph)) + 2 # CLS and SEP
                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation, paragraph,
                            self.discourse_label_types)
                        # validate_span_annotation(paragraph_annotation)
                        try:
                            span_indices = read_span_indices(
                                paragraph_annotation,
                                paragraph)
                        except:
                            continue
                        # span_BIO_labels = get_span_BIO_labels(span_indices, paragraph, tokenizer)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        # citation_BIO_labels = get_citation_BIO_labels(citation_mark_span_indices, paragraph, tokenizer)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)

                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        if dummy_discourse:
                            discourse_label_string = "".join(
                                [str(len(self.discourse_label_types)) for label
                                 in
                                 discourse_labels])
                        else:
                            discourse_label_string = "".join(
                                [str(self.discourse_label_types[label]) for
                                 label in
                                 discourse_labels])
                        if dummy_span:
                            # Put placeholder indices
                            span_label_string = "".join(
                                [str(len(self.span_label_types)) for label in
                                 span_BIO_labels])
                        else:
                            span_label_string = "".join(
                                [str(self.span_label_types[label]) for label in
                                 span_BIO_labels])

                        if dummy_citation:
                            citation_label_string = "".join(
                                [str(len(self.citation_label_types)) for label
                                 in
                                 citation_BIO_labels])
                        else:
                            citation_label_string = "".join(
                                [str(self.citation_label_types[label]) for label
                                 in
                                 citation_BIO_labels])

                        self.samples.append({
                            'id': paragraph_id,
                            'paragraph': paragraph.replace("[BOS]",
                                                           tokenizer.sep_token),
                            'discourse_label': discourse_label_string,
                            'span_label': span_label_string,
                            'citation_label': citation_label_string,
                            'N_tokens': N_tokens
                        })
                except:
                    continue

            else:
                for paragraph_id, paragraph in zip(paragraph_ids, paragraphs):
                    N_tokens = len(
                        tokenizer(paragraph, return_offsets_mapping=True)[
                            "offset_mapping"])
                    self.samples.append({
                        'id': paragraph_id,
                        'paragraph': paragraph.replace("[BOS]",
                                                       tokenizer.sep_token),
                        'N_tokens': N_tokens
                    })

    def update_samples(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)


class CORWAanalysisDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, context_window=2,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/selected_cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/selected_cited_pdf_parses.jsonl"
                 ):
        self.max_sent_len = MAX_SENT_LEN
        self.context_window = context_window
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

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        for text_file in text_files:
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                text_file, tokenizer, self.max_sent_len)

            paragraph_ids = []
            pi = 0
            for b in paragraph_begins:
                if b:
                    part_id = 0
                    paragraph_ids.append(
                        paper_id + "_" + str(pi) + "_" + str(part_id))
                    pi += 1
                else:
                    part_id += 1
                    paragraph_ids.append(
                        paper_id + "_" + str(pi - 1) + "_" + str(part_id))

            annotation_file = text_file.replace(".txt", ".ann")
            all_annotations = read_annotations(annotation_file, offsets)

            for paragraph_id, paragraph, paragraph_annotation in zip(paragraph_ids, paragraphs, all_annotations):
                for annotation in paragraph_annotation:
                    assert paragraph[annotation[0]:annotation[1]] == annotation[-1]
                tokens = tokenizer.tokenize(paragraph, add_special_tokens=True)
                # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                sentences = [sent for sent in paragraph.split("[BOS] ")[1:]]

                offset_mapping = \
                    tokenizer(paragraph, return_offsets_mapping=True)[
                        "offset_mapping"]
                N_tokens = len(offset_mapping)
                discourse_labels = read_discourse_labels(paragraph_annotation,
                                                         paragraph,
                                                         self.discourse_label_types)
                discourse_labels = ["Transition" if disc == "Intro" else disc
                                    for disc in discourse_labels]

                span_indices = read_span_indices(paragraph_annotation,
                                                 paragraph)
                span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                         offset_mapping)
                citation_mark_span_indices = read_citation_mark(
                    paragraph_annotation, paragraph)
                citation_BIO_labels = get_aligned_BIO_labels(
                    citation_mark_span_indices, offset_mapping)

                # print(tokenizer.tokenize(paragraph))
                assert (N_tokens == len(span_BIO_labels) == len(
                    citation_BIO_labels))
                # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                #    continue

                # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                    paragraph_id, sentences, self.related_work_jsons,
                    offset_mapping, citation_BIO_labels, separator="[BOS] ")
                paragraph_citation_links_pre = new_sentence_citation_link(
                    pargraph_citation_info, len(sentences))
                # sent2span_mapping, i_span = span_sentence_map(augmented_sentences)
                span_citation_mapping = map_span_citation(span_BIO_labels,
                                                          citation_BIO_labels,
                                                          pargraph_citation_info,
                                                          offset_mapping)

                sent2span_mapping, i_span = new_span_sentence_map(tokens,
                                                                  span_BIO_labels,
                                                                  bos="[BOS]")
                paragraph_citation_links = propagate_citation_cross_sentences(
                    sent2span_mapping, paragraph_citation_links_pre, i_span)
                
                span2sent_mapping = []
                for span_idx in range(len(span_citation_mapping)):
                    sent_ids = []
                    for sent_idx, sent in enumerate(sent2span_mapping):
                        if span_idx in sent:
                            sent_ids.append(sent_idx)
                    span2sent_mapping.append(sent_ids)
                
                #citation_type_by_sentence = []
                #for sent in sent2span_mapping:
                #    dominant_count = 0
                #    reference_count = 0
                #    for span_idx in sent:
                #        if span_citation_mapping[span_idx]["span_type"] == "Dominant":
                #            dominant_count += 1
                #        else:
                #            reference_count += 1
                #    citation_type_by_sentence.append((dominant_count, reference_count))
                
                #citation_type_by_sentence = citation_by_sentence(tokens, citation_BIO_labels)
                self.dataset.append({
                    "paragraph_id": paragraph_id,
                    "paragraph": paragraph,
                    # "related_work": augmented_paragraph,
                    "citation_links_by_sentence": paragraph_citation_links,
                    # "augmented_sentences": augmented_sentences,
                    "discourse_labels": discourse_labels,
                    "sentences": sentences,
                    "span_labels": span_BIO_labels,
                    "citation_labels": citation_BIO_labels,
                    "span_sent_mapping": sent2span_mapping,
                    "span2sent_mapping": span2sent_mapping,
                    # "tokens": tokens
                    # "i_span": i_span,
                    "span_citation_mapping": span_citation_mapping,
                    "offset_mapping": offset_mapping,
                    "citation_info": pargraph_citation_info,
                    #"citation_type_by_sentence": citation_type_by_sentence
                })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CitationSentenceGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(paragraph,
                                                                      span[
                                                                          "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context = paragraph_data[0]["range"], \
                                       paragraph_data[0]["cited_context"]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                        else:
                            result.append(
                                {"range": current, "cited_context": context})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]

                    result.append({"range": current, "cited_context": context})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class HumanEvaluationSpanGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation_sentence(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    sentence_citation_mapping = defaultdict(list)
                    for i_span, span in enumerate(span_citation_mapping):

                        # regex to find out the index of the citation mark
                        # maintain {(start, end) : {citaion: link}} --> convert to data

                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in span["span_citation_mapping"]["Dominant"]:
                                continue

                            citation_start, citation_end = span["span_citation_mapping"]["Dominant"][citation_pos_mark][0], \
                                                           span["span_citation_mapping"]["Dominant"][citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {citation_mark: " ".join(cited_context)})
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in \
                                    span["span_citation_mapping"]["Reference"]:
                                continue

                            citation_start, citation_end = \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][0], \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {citation_mark: " ".join(cited_context)})

                    citation_tracker = 0
                    for span_data in sentence_citation_mapping:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append(tokenizer.mask_token)
                        source.append(context_after)
                        for cited_context in sentence_citation_mapping[span_data]:
                            source.append(list(cited_context.values())[0])

                        if skip_no_citations and len(sentence_citation_mapping[span_data]) == 0:
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationSingleSentenceGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation_sentence(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    sentence_citation_mapping = defaultdict(list)
                    for i_span, span in enumerate(span_citation_mapping):

                        # regex to find out the index of the citation mark
                        # maintain {(start, end) : {citaion: link}} --> convert to data
                        cite_type = span["span_type"]
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in span["span_citation_mapping"]["Dominant"]:
                                continue

                            citation_start, citation_end = span["span_citation_mapping"]["Dominant"][citation_pos_mark][0], \
                                                           span["span_citation_mapping"]["Dominant"][citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {"citation_mark": " ".join(cited_context), "cite_type": cite_type})
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context = []
                            if citation_mark[-4:] == "_pos":
                                continue
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)

                            citation_pos_mark = citation_mark + "_pos"
                            if citation_pos_mark not in \
                                    span["span_citation_mapping"]["Reference"]:
                                continue

                            citation_start, citation_end = \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][0], \
                            span["span_citation_mapping"]["Reference"][
                                citation_pos_mark][1]
                            start = previous_sentence_end_index_from_span(
                                paragraph,
                                citation_start) + 1
                            end = next_sentence_start_index_from_span(paragraph,
                                                                      citation_end) - 1
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                            sentence_citation_mapping[(start, end)].append(
                                {"citation_mark": " ".join(cited_context), "cite_type": cite_type})

                    citation_tracker = 0
                    for span_data in sentence_citation_mapping:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append(tokenizer.mask_token)
                        source.append(context_after)

                        citations = []
                        for cited_context in sentence_citation_mapping[span_data]:
                            source.append(cited_context["citation_mark"])
                            citations.append(cited_context["cite_type"])
                        fin_citation = ""
                        if "Dominant" in citations:
                            fin_citation = "Dominant"
                        else:
                            fin_citation = "Reference"

                        if skip_no_citations and len(sentence_citation_mapping[span_data]) == 0:
                            continue
                        source = " ".join(source)
                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citation_label" : fin_citation
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class GenericCitationGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(
                                part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(
                            paragraph_annotation,
                            paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(
                            span_indices,
                            offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences,
                            self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(
                            paragraph,
                            span[
                                "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        source_data["citation_range"] = [span["char_start"],
                                                         span["char_end"]]

                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(
                            cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context, citation_ranges = paragraph_data[0][
                                                            "range"], \
                                                        paragraph_data[0][
                                                            "cited_context"], [
                                                            paragraph_data[
                                                                0][
                                                                "citation_range"]]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                            citation_ranges.append(
                                paragraph_data[i]["citation_range"])
                        else:
                            result.append(
                                {"range": current, "cited_context": context,
                                 "citation_ranges": citation_ranges})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]
                            citation_ranges = [
                                paragraph_data[i]["citation_range"]]

                    result.append(
                        {"range": current, "cited_context": context,
                         "citation_ranges": citation_ranges})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(
                                source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citations": [paragraph[d[0]:d[1]].replace(
                                    "[BOS] ", "") for d in
                                    span_data["citation_ranges"]]
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class GenericCitationGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True,
                 MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 skip_no_citations=False
                 ):

        def get_title_abstract(paper_dict):
            paras = [para["text"] for para in paper_dict["abstract"]]
            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(
                                part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph,
                                      return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(
                            paragraph_annotation,
                            paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(
                            span_indices,
                            offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences,
                            self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        print("Skip " + paragraph_id)

                    paragraph_data = []
                    for i_span, span in enumerate(span_citation_mapping):
                        source_data = {}
                        start = previous_sentence_end_index_from_span(
                            paragraph,
                            span[
                                "char_start"]) + 1
                        end = next_sentence_start_index_from_span(paragraph,
                                                                  span[
                                                                      "char_end"]) - 1

                        source_data["range"] = [start, end]
                        source_data["citation_range"] = [span["char_start"],
                                                         span["char_end"]]

                        cited_context = []
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Dominant"].items():
                            cited_context.append("[B_Dominant]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Dominant]")
                        for citation_mark, link in \
                                span["span_citation_mapping"][
                                    "Reference"].items():
                            cited_context.append("[B_Reference]")
                            cited_context.append(citation_mark)
                            cited_context.append(tokenizer.sep_token)
                            if link in self.cited_paper:
                                cited_context.append(
                                    get_title_abstract(
                                        self.cited_paper[link]))
                            cited_context.append("[E_Reference]")
                        source_data["cited_context"] = " ".join(
                            cited_context)
                        paragraph_data.append(source_data)

                    if len(paragraph_data) < 1:
                        continue
                    citation_tracker = 0
                    # Merge sentences having overlapped span
                    paragraph_data.sort(key=lambda x: x["range"][0])
                    result = []
                    current, context, citation_ranges = paragraph_data[0][
                                                            "range"], \
                                                        paragraph_data[0][
                                                            "cited_context"], [
                                                            paragraph_data[
                                                                0][
                                                                "citation_range"]]
                    for i in range(1, len(paragraph_data)):
                        if current[1] >= paragraph_data[i]["range"][0]:
                            current[1] = max(current[1],
                                             paragraph_data[i]["range"][1])
                            context = context + paragraph_data[i][
                                "cited_context"]
                            citation_ranges.append(
                                paragraph_data[i]["citation_range"])
                        else:
                            result.append(
                                {"range": current, "cited_context": context,
                                 "citation_ranges": citation_ranges})
                            context = paragraph_data[i]["cited_context"]
                            current = paragraph_data[i]["range"]
                            citation_ranges = [
                                paragraph_data[i]["citation_range"]]

                    result.append(
                        {"range": current, "cited_context": context,
                         "citation_ranges": citation_ranges})

                    for span_data in result:
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)

                        start, end = span_data["range"]

                        context_before = paragraph[:start].replace(
                            "[BOS] ", "")
                        context_after = paragraph[end:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 start:end].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        source.append("[Dominant]")
                        source.append(context_after)
                        source.append(span_data["cited_context"])

                        if skip_no_citations and not span_data[
                            "cited_context"].strip():
                            continue
                        source = " ".join(source)

                        if len(tokenizer.tokenize(
                                source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(
                                    citation_tracker),
                                "source": source,
                                "target": target,
                                "citations": [paragraph[d[0]:d[1]].replace(
                                    "[BOS] ", "") for d in
                                    span_data["citation_ranges"]]
                                # "full_target": " ".join(sentences)
                            })
                            citation_tracker += 1
                        else:
                            print("Context length exceeded "
                                  "than the maximum allowed: {}".format(
                                text_file))
            except Exception as e:
                print("Skip " + paper_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class CitationTextGenerationDataset(Dataset):
    def __init__(self, path_name: str, tokenizer, train=True, MAX_SENT_LEN=9999,
                 related_work_path='/home/data/XiangciLi/20200705v1/acl/related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl',
                 cited_paper_path="/home/data/XiangciLi/20200705v1/acl/cited_pdf_parses.jsonl",
                 citing_paper_path="/home/data/XiangciLi/20200705v1/acl/pdf_parses.jsonl",
                 include_intro=True,
                 include_conclusion=False,
                 conclusion_sections=None,
                 skip_no_citations=False
                 ):

        if conclusion_sections is None:
            # conclusion_sections = ['Conclusion', 'Conclusions',
            #                        'Conclusion and Future Work',
            #                        'Conclusions and Future Work',
            #                        'Conclusions and future work']
            conclusion_sections = ['Experimental Results',
                                   'Results',
                                   'Results and Discussion',
                                   'Results and Analysis',
                                   'Experiments and Results',
                                   'Main Results']

        def get_title_abstract(paper_dict, include_result=False):
            paras = [para["text"] for para in paper_dict["abstract"]]
            conclusions = []
            if include_result:
                for sec in paper_dict["body_text"]:
                    for conclusion_sec in conclusion_sections:
                        if sec["section"].lower() == conclusion_sec.lower():
                            conclusions.append(sec["text"])
                            break
            if len(conclusions) > 0:
                return paper_dict["title"] + " | " + " ".join(
                    paras) + " | " + " ".join(conclusions)

            return paper_dict["title"] + " | " + " ".join(paras)

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        text_files = glob(os.path.join(path_name, "*.txt"))
        if len(text_files) == 0:
            text_files = []
            paths = glob(os.path.join(path_name, "*"))
            for path in paths:
                text_files.extend(glob(os.path.join(path, "*.txt")))

        self.citing_paper = read_related_work_jsons(citing_paper_path)
        self.related_work_jsons = read_related_work_jsons(related_work_path)
        self.cited_metadata = read_related_work_jsons(cited_metadata_path)
        self.cited_paper = read_related_work_jsons(cited_paper_path)

        self.dataset = []
        self.samples = []
        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                if paper_id in self.citing_paper:
                    introduction_paragraphs = []
                    for para in self.citing_paper[paper_id]["body_text"]:
                        if "introduction" in para["section"].lower():
                            introduction_paragraphs.append(para["text"])
                    introduction_text = " ".join(introduction_paragraphs)
                else:
                    introduction_text = get_title_abstract(
                        self.related_work_jsons[paper_id])

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:
                        tokens = tokenizer.tokenize(paragraph,
                                                    add_special_tokens=True)
                        # sentences = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent)) for sent in paragraph.split("[BOS] ")[1:]]
                        sentences = [sent for sent in
                                     paragraph.split("[BOS] ")[1:]]

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)
                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        # if not N_tokens == len(span_BIO_labels) == len(citation_BIO_labels):
                        #    print(N_tokens, len(span_BIO_labels), len(citation_BIO_labels), paragraph_id, "Skipped")
                        #    continue

                        # augmented_paragraph, augmented_sentences = make_augmented_paragraphs(tokens, tokenizer, discourse_tokens, discourse_labels, span_BIO_labels, citation_BIO_labels)
                        # paragraph_citation_links_pre = new_sentence_citation_link(paragraph_id, augmented_sentences, self.related_work_jsons, tokenizer)
                        pargraph_citation_info = s2orc_to_corwa_paragraph_index(
                            paragraph_id, sentences, self.related_work_jsons,
                            offset_mapping, citation_BIO_labels,
                            separator="[BOS] ")
                        paragraph_citation_links_pre = new_sentence_citation_link(
                            pargraph_citation_info, len(sentences))
                        # span_sent_mapping, i_span = span_sentence_map(augmented_sentences)
                        span_citation_mapping = map_span_citation(
                            span_BIO_labels,
                            citation_BIO_labels,
                            pargraph_citation_info,
                            offset_mapping)
                        span_sent_mapping, i_span = new_span_sentence_map(
                            tokens,
                            span_BIO_labels,
                            bos="[BOS]")
                        paragraph_citation_links = propagate_citation_cross_sentences(
                            span_sent_mapping, paragraph_citation_links_pre,
                            i_span)
                        self.dataset.append({
                            "paragraph_id": paragraph_id,
                            "paragraph": paragraph,
                            # "related_work": augmented_paragraph,
                            "citation_links_by_sentence": paragraph_citation_links,
                            # "augmented_sentences": augmented_sentences,
                            "discourse_labels": discourse_labels,
                            "sentences": sentences,
                            # "span_labels": span_BIO_labels,
                            # "citation_labels": citation_BIO_labels,
                            "span_sent_mapping": span_sent_mapping,
                            # "tokens": tokens
                            # "i_span": i_span,
                            "span_citation_mapping": span_citation_mapping,
                            # "offset_mapping": offset_mapping,
                            "citation_info": pargraph_citation_info
                        })
                    except:
                        pass
                        #print("Skip " + paragraph_id)

                    for i_span, span in enumerate(span_citation_mapping):
                        citation_marks = []
                        source = []
                        if include_intro:
                            source.append(introduction_text)
                            source.append(tokenizer.sep_token)
                        context_before = paragraph[:span["char_start"]].replace(
                            "[BOS] ", "")
                        context_after = paragraph[span["char_end"]:].replace(
                            "[BOS] ", "")
                        target = paragraph[
                                 span["char_start"]:span["char_end"]].replace(
                            "[BOS] ", "")
                        source.append(context_before)
                        if len(span["span_citation_mapping"]["Dominant"]) > 0:
                            source.append("[Dominant]")
                        else:
                            source.append("[Reference]")
                        source.append(context_after)

                        citation_marks_flag = False
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Dominant"].items():
                            source.append("[B_Dominant]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link],
                                                       include_result=include_conclusion))
                            source.append("[E_Dominant]")
                        for citation_mark, link in \
                        span["span_citation_mapping"][
                            "Reference"].items():
                            source.append("[B_Reference]")
                            source.append(citation_mark)
                            source.append(tokenizer.sep_token)
                            citation_marks.append(citation_mark)
                            citation_marks_flag = True
                            if link in self.cited_paper:
                                source.append(
                                    get_title_abstract(self.cited_paper[link]))
                            source.append("[E_Reference]")
                        source = " ".join(source)
                        if skip_no_citations and not citation_marks_flag:
                            continue

                        if len(tokenizer.tokenize(source)) <= self.max_sent_len:
                            self.samples.append({
                                "id": paragraph_id + "_" + str(i_span),
                                "source": source,
                                "target": target,
                                # "full_target": " ".join(sentences)
                                #"citations": "#".join(citation_marks)
                            })
            except:
                #print("Skip "+paper_id)
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def merge(self, dataset):
        self.samples.extend(dataset.samples)
        self.dataset.extend(dataset.dataset)

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class JointTaggerAgreementDataset(Dataset):
    def __init__(self, text_files: str, tokenizer, MAX_SENT_LEN=512):

        self.max_sent_len = MAX_SENT_LEN
        self.discourse_label_types = {"Intro": 0,
                                      "Single_summ": 1,
                                      "Multi_summ": 2,
                                      "Narrative_cite": 3,
                                      "Reflection": 4,
                                      "Transition": 5,
                                      "Other": 6
                                      }

        discourse_tokens = []
        for k, v in self.discourse_label_types.items():
            discourse_tokens.append("[" + k + "]")

        self.all_discourse_labels, self.all_spans_BIO_labels, self.all_citations_BIO_labels = [], [], []


        for text_file in tqdm(text_files):
            head, tail = os.path.split(text_file)
            paper_id = tail.split(".")[0]
            try:
                paragraphs, offsets, paragraph_begins = read_paragraphs_split(
                    text_file, tokenizer, self.max_sent_len)

                paragraph_ids = []
                pi = 0
                for b in paragraph_begins:
                    if b:
                        part_id = 0
                        paragraph_ids.append(
                            paper_id + "_" + str(pi) + "_" + str(part_id))
                        pi += 1
                    else:
                        part_id += 1
                        paragraph_ids.append(
                            paper_id + "_" + str(pi - 1) + "_" + str(part_id))

                annotation_file = text_file.replace(".txt", ".ann")
                all_annotations = read_annotations(annotation_file, offsets)

                for paragraph_id, paragraph, paragraph_annotation in zip(
                        paragraph_ids, paragraphs, all_annotations):
                    for annotation in paragraph_annotation:
                        assert paragraph[annotation[0]:annotation[1]] == \
                               annotation[
                                   -1]
                    try:

                        offset_mapping = \
                            tokenizer(paragraph, return_offsets_mapping=True)[
                                "offset_mapping"]
                        N_tokens = len(offset_mapping)
                        discourse_labels = read_discourse_labels(
                            paragraph_annotation,
                            paragraph,
                            self.discourse_label_types)

                        # NEEDED
                        span_indices = read_span_indices(paragraph_annotation,
                                                         paragraph)

                        span_BIO_labels = get_aligned_BIO_labels(span_indices,
                                                                 offset_mapping)
                        citation_mark_span_indices = read_citation_mark(
                            paragraph_annotation, paragraph)
                        citation_BIO_labels = get_aligned_BIO_labels(
                            citation_mark_span_indices, offset_mapping)
                        # print(tokenizer.tokenize(paragraph))
                        citation_BIO_labels = [x[2:] if x.startswith("B_") or x.startswith("I_") else x for x in citation_BIO_labels ]
                        span_BIO_labels = [x[2:] if x.startswith("B_") or x.startswith("I_") else x for x in span_BIO_labels ]
                        assert (N_tokens == len(span_BIO_labels) == len(
                            citation_BIO_labels))
                        self.all_discourse_labels.extend(discourse_labels)
                        self.all_citations_BIO_labels.extend(citation_BIO_labels)
                        self.all_spans_BIO_labels.extend(span_BIO_labels)
                    except:
                        print("Skip " + paragraph_id)
            except:
                print("Skip " + paper_id)


class LEDRelatedWorkMLMDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, path_name, train, bos_token="<s>", eos_token="</s>"):
        self.samples = []

        with open(path_name, "r") as f_pdf:
            for line in tqdm(f_pdf):
                related_work_dict = json.loads(line)
                year = related_work_dict["year"]
                if year is None:
                    year = 0
                if (train and year <= 2017) or (not train and year == 2018):
                    # test set should include papers publised in 2019 and later
                    for pi, para in enumerate(
                            related_work_dict["related_work"]):
                        source, target = makeMLMsample(para["text"], "<mask>")
                        self.samples.append({
                            'id': related_work_dict["paper_id"] + "_" + str(pi),
                            'source': " ".join([bos_token, source, eos_token]),
                            'target': target
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset


class SimpleCrossDocumentLMdataset(Dataset):
    def __init__(self, path_name, tokenizer, train=True,
                 MAX_SENT_LEN=16000, bos_token="<s>", eos_token="</s>",
                 mask_token="<mask>",
                 bod_token="<doc>", eod_token="</doc>",
                 # related_work_path = '/home/data/XiangciLi/20200705v1/acl/selected_related_work.jsonl',
                 cited_metadata_path='/home/data/XiangciLi/20200705v1/acl/cited_metadata.jsonl'):
        self.max_sent_len = MAX_SENT_LEN
        self.related_work_jsons = read_related_work_jsons(path_name)
        self.cited_metadata_jsons = read_related_work_jsons(cited_metadata_path)

        self.samples = []

        for i, (ID, related_work) in tqdm(
                enumerate(self.related_work_jsons.items())):
            year = related_work["year"]
            if year is None:
                year = 0
            if (train and year <= 2017) or (not train and year == 2018):
                bib_entries = related_work["bib_entries"]
                for paragraph in related_work["related_work"]:
                    inputs = []
                    noisy_text, target = makeMLMsample(paragraph["text"])
                    inputs.extend([bod_token, noisy_text, eod_token])
                    if len(tokenizer(target)["input_ids"]) > self.max_sent_len:
                        continue
                    source = " ".join(inputs)

                    for citation in paragraph["cite_spans"]:
                        if citation["ref_id"] in bib_entries:
                            reference_link = bib_entries[citation["ref_id"]][
                                "link"]
                            if reference_link in self.cited_metadata_jsons:
                                cited_metadata = self.cited_metadata_jsons[
                                    reference_link]
                                title = cited_metadata["title"]
                                if title is None:
                                    title = ""
                                abstract = cited_metadata["abstract"]
                                if abstract is None:
                                    abstract = ""
                                inputs.extend(
                                    [bod_token, title, tokenizer.sep_token,
                                     abstract, eod_token])
                                if len(tokenizer(" ".join(inputs))[
                                           "input_ids"]) > self.max_sent_len:
                                    break
                                source = " ".join(inputs)
                    self.samples.append({
                        "id": ID + "_" + str(i),
                        "source": source,
                        "target": target
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_dict(self):
        dataset = {}
        keys = self.samples[0].keys()
        for k in keys:
            dataset[k] = []
        for sample in self.samples:
            for k in keys:
                dataset[k].append(sample[k])
        return dataset
