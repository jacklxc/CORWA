# CORWA
This work is just accepted by NAACL 2022. A short [video](https://www.youtube.com/watch?v=ervPq7eAC9o) for the previous SciNLP 2021 workshop is available.

## Environment
The required packages are in `requirements.txt`. Note that transformers==4.2.0 is strongly recommended.

## Overall data pipeline
We extract related work sections from S2ORC dataset, whose metadata attributes containing ACL id, to `related_work.jsonl`. Then for the annotations and experiments in CORWA paper, we generally store the related work sections in the BRAT format, then read from, and write to BRAT format. However, if you are simply interested in using the joint related work tagger, you may use `pipeline/pipeline.py` to start from `related_work.jsonl` and output a jsonl file.

### View the dataset in BRAT
* Download BRAT from https://brat.nlplab.org/index.html to view the annotation.
* The annotation (.ann) file is a charater-level mapping to txt files.

## Dataset
* The CORWA train- and test-set are under `data/`.

## Checkpoints
* [Joint related work tagger](https://drive.google.com/file/d/1pE1J1MK5D2U7oxAwqdwWNgKnoi1wTp0T/view?usp=sharing)
* LED citation span generator

## Experiments
Under `experiments/`:
* cross_validation_joint_tagger.py, train_joint_tagger.py and predict_joint_tagger.py cross-validate, train and predict the joint related work tagger, described in section 4.
* pattern_extraction.py corresponds to the experiments for Table 7 and 8.
* writing_style.ipynb contains the code for the statistics and visualizations in section 3.3.
* rouge_salient_sentence contains the code for the experiment in section 5.1.
* related_work2brat.ipynb shows how to convert related work sections extracted from S2ORC's pdf_parse json objects to BRAT format for annotation.
* LED.py, LED_sentence_generation.py are the code for citation span generation and sentence-level baseline, described in section 5.2.

## Usage of joint related work tagger for future work
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
Coming to arXiv soon.
