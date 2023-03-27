import argparse
import pathlib
import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

LIMIT = 200

def extract_candidates(text: str, threshold=4):
    GRAMMAR_EN = """  NP:
{<NN.*|JJ>*<NN.*>}"""   # Adjective(s)(optional) + Noun(s)
    keyphrase_candidate = set()
    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    tag = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    trees = np_parser.parse_sents(tag)  # Generator with one tree per sentence

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, _ in subtree.leaves()))
    
    keyphrase_candidate = [kp for kp in keyphrase_candidate if len(kp.split()) <= threshold]
    return keyphrase_candidate

def stem(predictions: list, remove_subphrase=True, threshold=4):
    porter_stemmer = PorterStemmer()
    new_pred = set()
    for e in predictions:
        e = e.replace('-',' ')
        e = e.split(' ')
        if len(e) > threshold:
            continue
        stemmed = [porter_stemmer.stem(q) for q in e]
        c = ' '.join(stemmed)
        new_pred.add(c.strip())
    if remove_subphrase:
        for e in new_pred:
            flg = 0 
            for w in new_pred:
                if w in e:
                    flg = 1
                    break
            if flg == 1:
                new_pred.remove(e)
    return list(new_pred)

def extract_keyphrases(dataset_name: str, split: str, output_path: str):
    dataset_path = f"data/{dataset_name}"

    phrase_dic = {}
    with open(f"{dataset_path}/{split}.json", "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == LIMIT:
                break
            data = json.loads(line)
            if "document" in data:
                doc = " ".join(data["document"])
                phrase_dic[i] = extract_candidates(doc.lower())
            else:
                #ldkp3k
                section_list = data["sections"]
                section_text_list = data["sec_text"]
                if len(section_list) != len(section_text_list):
                    raise(f"section length != section_text length: ", i)
                for j, section_text in enumerate(section_text_list):
                    phrase_dic[f"{i}_{j}"] = extract_candidates(" ".join(section_text).lower())

    path_safe_name = dataset_name.replace("/", "-")
    with open(f"{output_path}/{path_safe_name}-{split}-phrases.json", "w") as f:
        json.dump(phrase_dic, f)


if __name__ == "__main__":
    # Example: python3 02-extract_keyphrases.py --dataset midas/kp20k --split train
    # Example2: python3 02-extract_keyphrases.py --dataset midas/ldkp3k --split train
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default="temp")

    args = parser.parse_args()
    # Get all the variables
    dataset_name = args.dataset
    split = args.split
    output_path = args.output

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('punkt')

    extract_keyphrases(dataset_name, split, output_path)
