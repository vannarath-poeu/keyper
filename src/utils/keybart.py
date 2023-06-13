
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Text2TextGenerationPipeline

class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model_name: str, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]

def keybart(generator, doc, top_n=10):
    keyphrases = generator(doc)[0][:top_n]
    return keyphrases

def keybart_list(generator, doc_list, top_n=10):
    keyphrase_list = generator(doc_list)
    return [ k[:top_n]  for k in keyphrase_list]