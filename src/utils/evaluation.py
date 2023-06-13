def evaluate(top_N_keyphrases, references):
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases) if len(top_N_keyphrases) > 0 else 0
    R = len(set(top_N_keyphrases) & set(references)) / len(references) if len(references) > 0 else 0
    F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
    return (P, R, F)

def generate_results():
    pass