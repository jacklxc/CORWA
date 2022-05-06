import re
from nltk.tokenize import sent_tokenize, word_tokenize

def patch_sent_tokenize(sentences):
    out = []
    i = 0
    while i < len(sentences):
        if i>0 and sentences[i-1][-4:] == " et." and sentences[i][:2] == "al":
            out[-1] += " " + sentences[i]
        elif i>0 and (sentences[i-1][-4:] == " al." or sentences[i-1]=="al."):
            out[-1] += " " + sentences[i]
        elif i>0 and sentences[i-1][-4:] == "e.g.":
            out[-1] += " " + sentences[i]
        elif i>0 and sentences[i-1][-4:] == "i.e.":
            out[-1] += " " + sentences[i]
        else:
            out.append(sentences[i])
        i += 1
    return out

def scientific_sent_tokenize(paragraph, add_bos_token=True, bos_token="[BOS]"):
    sentences = []
    for si, sentence in enumerate(patch_sent_tokenize(sent_tokenize(paragraph))):
        sentence = re.sub("([^\x00-\x7F])+","",sentence)
        if add_bos_token:
            sentence = bos_token + " " + sentence
        sentences.append(sentence)
    return sentences

