# CORWA
This work is accepted by NAACL 2022 ([video](https://youtu.be/9siOUrqlXXE)).

## Environment
The required packages are in `requirements.txt`. Note that transformers==4.2.0 is strongly recommended for Longformer-Encoder-Decoder citation span generator.

## Overall data pipeline
We extract related work sections from S2ORC dataset, whose metadata attributes containing ACL id, to `related_work.jsonl`. Then for the annotations and experiments in CORWA paper, we generally store the related work sections in the BRAT format, then read from, and write to BRAT format. However, if you are simply interested in using the joint related work tagger, you may use `pipeline/pipeline.py` to start from `related_work.jsonl` and output a jsonl file.

### View the dataset in BRAT
* Download BRAT from https://brat.nlplab.org/index.html to view the annotation.
* The annotation (.ann) file is a charater-level mapping to txt files.

## Dataset
* The CORWA train- and test-set are under `data/`.
* NEW: CORWA distant-set is also released under `data/`.

## Checkpoints
* [Joint related work tagger](https://drive.google.com/file/d/1pE1J1MK5D2U7oxAwqdwWNgKnoi1wTp0T/view?usp=sharing)
* [LED citation span generator](https://drive.google.com/file/d/1KX-rSo4xwS3wn-KY7FckHRCOeQDqP6p9/view?usp=sharing)

## Experiments
Under `experiments/`:

Implementations of the experiments in the CORWA paper. They are not necessarily recommended to reuse for the future work. Consult the next section for the future usage of joint related work tagger.

* `cross_validation_joint_tagger.py`, `train_joint_tagger.py` and `predict_joint_tagger.py` cross-validate, train and predict the joint related work tagger, described in section 4.
* `pattern_extraction.py` corresponds to the experiments for Table 7 and 8.
* `writing_style.ipynb` contains the code for the statistics and visualizations in section 3.3.
* `rouge_salient_sentence.ipynb` contains the code for the experiment in section 5.1.
* `related_work2brat.ipynb` shows how to convert related work sections extracted from S2ORC's pdf_parse json objects to BRAT format for annotation.
* `LED.py`, `LED_sentence_generation.py` are the code for citation span generation and sentence-level baseline, described in section 5.2.

## Usage of joint related work tagger for future work
Note that you may use newer package versions as specified in `requirements.txt` for joint related work tagger.

Under `pipeline/`:
* `filter_ACL_S2ORC.py` and `S2ORC_parse_related_work.ipynb` are example code to extract related work sections from S2ORC metadata and pdf_parses.
* Run `pipeline.py --related_work_file RELATED_WORK.FILE --checkpoint CHECKPOINT.FILE --output_file YOUR_OUTPUT.jsonl` to directly tag related work sections, and outputs a human-readable jsonl file.

* Output jsonl format:
    * Each line is a json object, corresponding to a paragraph of related work section.
    * Attributs:
    ```
    {
        "id": paragraph_id, // S2ORC_papar_id + paragraph_index
        "paragraph": string, // Paragraph of related work, with each sentence led by [BOS] token
        "discourse_tags": [string], // list of discourse tags for each sentence
        "span_citation_mapping": [ // Each element corresponds to a citation span.
            {
                "token_start": int,
                "token_end": int, // start and end indices of the citation span, in terms of tokens, tokenized by the SciBERT tokenizer by Huggingface
                "char_start": int,
                "char_end": int, // start and end indices of the citation span, in terms of characters
                "span_type": "Dominant" or "Reference",
                "span_citation_mapping": {
                    "Dominant": {
                        citation_mark: cited_paper_id
                    },
                    "Reference": {
                        citation_mark: cited_paper_id
                    },
                }
            }
        ]
    }
    ```

## Cite our paper
```
@inproceedings{li-etal-2022-corwa,
    title = "{CORWA}: A Citation-Oriented Related Work Annotation Dataset",
    author = "Li, Xiangci  and
      Mandal, Biswadip  and
      Ouyang, Jessica",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.397",
    pages = "5426--5440",
    abstract = "Academic research is an exploratory activity to discover new solutions to problems. By this nature, academic research works perform literature reviews to distinguish their novelties from prior work. In natural language processing, this literature review is usually conducted under the {``}Related Work{''} section. The task of related work generation aims to automatically generate the related work section given the rest of the research paper and a list of papers to cite. Prior work on this task has focused on the sentence as the basic unit of generation, neglecting the fact that related work sections consist of variable length text fragments derived from different information sources. As a first step toward a linguistically-motivated related work generation framework, we present a Citation Oriented Related Work Annotation (CORWA) dataset that labels different types of citation text fragments from different information sources. We train a strong baseline model that automatically tags the CORWA labels on massive unlabeled related work section texts. We further suggest a novel framework for human-in-the-loop, iterative, abstractive related work generation.",
}
```
