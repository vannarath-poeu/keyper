from nltk.stem.porter import PorterStemmer

def stem_keywords(keyword_list: list):
    porter_stemmer = PorterStemmer()
    stemmed_keywords = []
    for kw in keyword_list:
        stemmed = [porter_stemmer.stem(q) for q in kw.split()]
        stemmed_keywords.append(" ".join(stemmed))
    return stemmed_keywords