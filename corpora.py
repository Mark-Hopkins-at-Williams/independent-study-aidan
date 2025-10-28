import random
from typing import Dict, Tuple, List, Optional, Iterator, Callable
from torch.utils.data import DataLoader, IterableDataset

from tokenization import Tokenizer

CorpusId = Tuple[str, str] # typedef

class MultifileBitext(IterableDataset):
    def __init__(self, lang1_files: List[str], lang2_files: List[str], lines: Optional[List[Tuple[int, int]]] = None):
        self.lang1_files = lang1_files
        self.lang2_files = lang2_files
        self.lines = lines        
        
    def line_streamer(self, lang_index) -> Iterator[str]:
        lang_files = self.lang1_files if lang_index == 0 else self.lang2_files
        for file_index in range(len(self.lang1_files)):
            file_path = lang_files[file_index]
            current_line = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:  
                    if self.lines is None or self.lines[file_index][0] <= current_line < self.lines[file_index][1]:       
                        yield line.rstrip("\n")                
                    current_line += 1
                    if self.lines is not None and current_line >= self.lines[file_index][1]:
                        break

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        
        return zip(
            self.line_streamer(0), self.line_streamer(1)
        )


class Bitext(IterableDataset):
    def __init__(self, lang1_file: str, lang2_file: str, lines: Optional[Tuple[int, int]] = None):
        self.lang1_file = lang1_file
        self.lang2_file = lang2_file
        self.lines = lines        
        
    def line_streamer(self, file_path: str) -> Iterator[str]:
        current_line = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:  
                if self.lines is None or self.lines[0] <= current_line < self.lines[1]:       
                    yield line.rstrip("\n")                
                current_line += 1
                if self.lines is not None and current_line >= self.lines[1]:
                    break

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(
            self.line_streamer(self.lang1_file), self.line_streamer(self.lang2_file)
        )


class MixtureOfBitexts:
    def __init__(
        self,
        bitexts: Dict[Tuple[str, str], Bitext],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False
    ):
        self.bitexts = bitexts        
        self.keys = list(bitexts)
        self.batch_size = batch_size
        self.batch_iters = {}

        for key in self.keys:
            self.batch_iters[key] = self._create_iterator(key)

        total = sum(sampling_probs) if sampling_probs else len(bitexts)
        self.sampling_probs = [
            p / total for p in (sampling_probs or [1.0] * len(bitexts))
        ]
        
        self.only_once_thru = only_once_thru
        self.completed_bitexts = set()

    def _create_iterator(
        self, key: Tuple[str, str]
    ) -> Iterator[Tuple[List[str], List[str]]]:
        return iter(
            DataLoader(
                self.bitexts[key],
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
            )
        )

    def next_batch(self) -> Optional[Tuple[List[str], List[str], str, str]]:
        still_choosing = True
        while still_choosing and len(self.completed_bitexts) < len(self.keys):
            lang_pair = random.choices(self.keys, weights=self.sampling_probs, k=1)[0]
            try:
                lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])                
                still_choosing = False
            except StopIteration:
                if self.only_once_thru:
                    self.completed_bitexts.add(lang_pair)
                else: # start a new iterator for the chosen bitext
                    self.batch_iters[lang_pair] = self._create_iterator(lang_pair)                    
        if still_choosing:
            return None
        else:
            return lang1_sents, lang2_sents, lang_pair[0], lang_pair[1]

    @staticmethod
    def create_from_files(
        text_files: Dict[str, str],
        lps: List[Tuple[str, str, Optional[Tuple[int, int]]]],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None,
        only_once_thru: bool = False
    ) -> "MixtureOfBitexts":
        bitexts = {(l1, l2): Bitext(text_files[l1], text_files[l2], lines) for (l1, l2, lines) in lps}        
        return MixtureOfBitexts(bitexts, batch_size, sampling_probs, only_once_thru)
    
    @staticmethod
    def create_from_config(config: dict, split: str, only_once_thru: bool = False) -> "MixtureOfBitexts":
        all_corpora = dict()
        for corpus in config['corpora']:
            for key in config['corpora'][corpus]:
                all_corpora[(corpus, key)] = config['corpora'][corpus][key][split]
        bitexts = dict()
        for bitext in config['bitexts']:
            src = (bitext['corpus'], bitext['src'])
            tgt = (bitext['corpus'], bitext['tgt'])
            lines = bitext["train_lines"] if split == "train" else None
            bitexts[(src, tgt)] = Bitext(all_corpora[src], all_corpora[tgt], lines)
        params = config["finetuning_parameters"]
        return MixtureOfBitexts(bitexts, params['batch_size'], sampling_probs=None, only_once_thru=only_once_thru)
        

    def get_language_codes(self) -> List[str]:
        return sorted({code for pair in self.keys for code in pair})


class TokenizedMixtureOfBitexts:
    def __init__(
        self,
        mixture_of_bitexts: MixtureOfBitexts,
        tokenizer: Tokenizer,
        lang_codes: Dict[CorpusId, str],
        permutation_map: Dict[CorpusId, Callable[[int], int]] = dict()
    ):
        self.mixture_of_bitexts = mixture_of_bitexts
        self.tokenizer = tokenizer
        self.lang_codes = lang_codes
        self.permutation_map = permutation_map

    def _tokenize(self, sents: List[str], corpus: CorpusId, alt_pad_token: int = None):
        tokens = self.tokenizer(sents, lang_code = self.lang_codes[corpus])
        if alt_pad_token is not None:
            pad_token_id = self.tokenizer.get_special_tokens()['<pad>']
            tokens.input_ids[tokens.input_ids == pad_token_id] = alt_pad_token            
        if corpus in self.permutation_map: # apply the permutation
            p = self.permutation_map[corpus]
            tokens.input_ids.apply_(p) # modifies in-place
        return tokens

    def next_batch(self):
        batch = self.mixture_of_bitexts.next_batch()
        if batch is None:
            return None
        lang1_sents, lang2_sents, lang1, lang2 = batch
        lang1_tokenized = self._tokenize(lang1_sents, lang1)
        lang2_tokenized = self._tokenize(lang2_sents, lang2, alt_pad_token=-100)
        return lang1_tokenized, lang2_tokenized, lang1, lang2
