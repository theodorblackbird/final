from preprocess.phonem import BasePhonem
import gruut


class gruutPhonem(BasePhonem):

    def _translate(self, text):

        text_wo_punc, punc, pos = self._preprocess(text)

        sentence = []
        for x in text_wo_punc :
            for sent in gruut.sentences(x, lang="en-us"):
                part = []
                for word in sent :
                    w = "".join(word.phonemes)
                    part.append(w)
                part = " ".join(part)
                sentence.append(part)
        return "".join(self._set_back_punc(sentence, punc, pos))



if __name__ == "__main__" :
    phon = gruutPhonem()
    a = phon._translate("This, is a test ! One should try, \"at least\" one time !")
    print(a)
