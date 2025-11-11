import json
import random
from tqdm import tqdm
from typing import Dict

class iterativeBackTranslation():
    def __init__(self, lang_to_path: Dict[str, str], model_dir: str, lang_codes: Dict[str, str]) -> None:
        self.langs_to_path = lang_to_path
        self.data_dir = model_dir
        self.lang_codes = lang_codes
    def translate(
        self, 
        src_tokenized,
        tokenizer,
        model,
        tgt_lang,
        permutation=None,
        a=32,
        b=3,
        num_beams=4,
        **kwargs
    ):
        model.eval()
        result = model.generate(
            **src_tokenized.to(model.device),
            forced_bos_token_id=tokenizer.get_special_tokens()[tgt_lang],
            max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
            num_beams=num_beams,
            **kwargs
        )
        result = result.to('cpu')
        if permutation is not None:
            result.apply_(permutation.get_inverse())
        return tokenizer.batch_decode(result)

    def backTranslate(self, in_model, target_langauges, tokenizer):
        ret = {}
        for lang in target_langauges:
            translation_len, lines = self.getMonoSample(lang)
            start = random.randint(0, len(lines)-translation_len)
            translation_lines = lines[start:start+translation_len]

            with open(f"{self.data_dir}/{lang}/trainBackTrans.es","w") as es, \
                open(f"{self.data_dir}/{lang}/trainBackTrans.tsv","w") as tgt:
                    for line in tqdm(translation_lines,desc=f"Translating: {lang}", total=len(translation_lines)):
                        inputs = tokenizer([line])
                        inputs = inputs.to("cuda")
                        translated_tokens = self.translate(src_tokenized=inputs, model=in_model,tokenizer=tokenizer, tgt_lang=self.lang_codes[lang])
                        es.write(line + "\n")
                        tgt.write(translated_tokens[0] + "\n")

    def getMonoSample(self, target_language):
        mono = []
        print(self.langs_to_path)
        print(target_language)
        for lang, paths in self.langs_to_path.items():
            if lang != target_language:
                for file in paths:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            sentences = [line.strip() for line in f if line.strip()]
                            mono.extend(sentences)
                    except FileNotFoundError:
                        print(f"Warning: File not found: {file}")
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
            else:
                print("here")
                for file in paths:
                    with open(file, 'r', encoding='utf-8') as f:
                            translate_len = sum(1 for line in f if line.strip())
        return translate_len, mono
                            
