import glob
import os
import sys

from sklearn.metrics import cohen_kappa_score, accuracy_score
from transformers import AutoTokenizer

from dataset import JointTaggerAgreementDataset


def get_cross_validation_file(text_file, cross_val_dir):
    """Get the cross validation file from different annotator"""
    val_file = os.path.join(
        cross_val_dir,
        os.path.split(text_file)[1]
    )
    if os.path.exists(val_file):
        if not os.path.exists(val_file.replace(".txt", ".ann")):
            print(
                "Text file exists, but couldn't find annotation file for {}".format(
                    val_file))
        else:
            return val_file
    return ""


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
tokenizer.add_special_tokens(additional_special_tokens)

files = glob.glob(sys.argv[1]+ "*.txt")
cross_dir = sys.argv[2]

ann_1_files, ann_2_files = [], []
for file in files:
    cross_file = get_cross_validation_file(
        file, cross_dir
    )
    if cross_file:
        ann_1_files.append(file)
        ann_2_files.append(cross_file)

print("Number of common files: {}".format(len(ann_1_files)))

data_one = JointTaggerAgreementDataset(
    ann_1_files, tokenizer
)
data_two = JointTaggerAgreementDataset(
    ann_2_files, tokenizer
)

discourse_kappa = cohen_kappa_score(
    data_one.all_discourse_labels,
    data_two.all_discourse_labels
)
discourse_acc = accuracy_score(
    data_one.all_discourse_labels,
    data_two.all_discourse_labels
)
print("Discourse cohen kappa {}".format(discourse_kappa))
print("Discourse accuracy {}".format(discourse_acc))


citation_kapp = cohen_kappa_score(
    data_one.all_citations_BIO_labels,
    data_two.all_citations_BIO_labels
)

print("Citation cohen kappa {}".format(citation_kapp))


span_kappa = cohen_kappa_score(
    data_one.all_spans_BIO_labels,
    data_two.all_spans_BIO_labels
)

print("Span cohen kappa {}".format(span_kappa))

