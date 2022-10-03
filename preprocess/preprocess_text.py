import gruut
from tqdm import tqdm
from model.tokenizer import _punctuations

if __name__ == "__main__":
    with open("preprocess/metadata_pp.csv", 'w') as o:
        with open('preprocess/metadata.csv') as f:
            for row in tqdm(f):
                t = row.strip().split("|")
                words = []
                for x in gruut.sentences(t[2], lang="en-us") :
                    for word in x:
                        if word.phonemes:
                            if word.text in _punctuations:
                                words.append(word.text)
                            else:
                                words.append("".join((word.phonemes)))
                o.write(t[0] + '|' + " ".join(words) + '\n')
