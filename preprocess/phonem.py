import abc
import re

_DEF_PUNCS = ';:,.!?¡¿—…"«»“”'

class BasePhonem(abc.ABC):
    """ Base class for char. to phon. translation """

    def __init__(self, language="english"):
        self._language = language
        self._punc_regex = re.compile(rf"(\s*[{re.escape(_DEF_PUNCS)}]+\s*)+")
    
    def _preprocess(self, text):
        text = text.strip()
        text, punc, pos = self._split_punc(text)
        return text, punc, pos

    def _postprocess(self, phon, punc, pos):
        return self._set_back_punc(phon, punc, pos)
    
    def _split_punc(self, text):
        punc = re.findall(self._punc_regex, text) #punc.
        split = re.split(self._punc_regex, text) #split at punc. positions
        split = list(filter(None, split)) #filter empty strings
        pos = [True if x.strip() in list(_DEF_PUNCS) else False for x in split] #punc. map positions
        text = [split[i[0]] for i in enumerate(pos) if not(i[1])] #text w/o punc.
        return text, punc, pos
    def _set_back_punc(self, phon, punc, pos):
        phon_w_punc = []
        i_phon = 0
        i_punc = 0
        for p in pos :
            if p:
                phon_w_punc.append(punc[i_punc])
                i_punc += 1
            else:
                phon_w_punc.append(phon[i_phon])
                i_phon += 1
        return phon_w_punc


    @abc.abstractmethod
    def _translate(self, text):
        return
