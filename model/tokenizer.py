

#Come from CoquiTTS

# DEFAULT SET OF GRAPHEMES
_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_punctuations = "!'(),-.:;? "


# DEFAULT SET OF IPA PHONEMES
# Phonemes definition (All IPA characters)
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧʲ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


class Tokenizer:

    def __init__(self, beos_token=True, vocab=None):
        self.beos_token = beos_token
        if vocab == None:
            self.vocab = list(_phonemes + _characters + _punctuations) + [_pad] + [_eos] + [_bos]
            self.id2char = {c: i for c,i in enumerate(self.vocab)}
            self.char2id = {v: k for k, v in self.id2char.items()}
        else:
            self.vocab = vocab



    def encode(self, X : str):
        res = []
        for x in X :
            try:
                z = self.char2id[x]
            except KeyError:
                print(f"Unknown character \"{x}\", discarding it ...")
                z = -1;
            if z != -1:
                res.append(z)
        return res


    def decode(self, Y : list[int]):
        res = ""
        for y in Y :
            try:
                z = self.id2char[y]
            except KeyError:
                print(f"ID \"{y}\" not in range, discarding it ...")
                z = ""
            res += z
        return res


if __name__ == "__main__":
    from preprocess.gruut_phonem import gruutPhonem
    tokenizer = Tokenizer()

    print(tokenizer.decode(tokenizer.encode("hello world @ cɟkɡqɢ")))
