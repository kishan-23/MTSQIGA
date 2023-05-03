import re
import string
import itertools
import spacy
import nltk
from nltk.data import load
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from truecase import get_true_case

nltk.download('stopwords')

# Using the stopwords.

# Initialize the stopwords
stoplist = stopwords.words('english')


class Preprocessor:

    def __init__(self, *args):
        self.sentencesNum = 0
        self.numOfWords = 0

    def preprocessing_text(self, text):
        tokenizer = load('tokenizers/punkt/english.pickle')

        tokenized_sents = tokenizer.tokenize(text)
#         print(tokenized_sents)

        # Part-of-speech tagging of text sentences
        text_tagged = self.part_of_speech_tagging(tokenized_sents)
#         print(text_tagged)

#         working_sentence = [sent.lower() for sent in tokenized_sents] #****!!!!! lower after tokenization
#         print(working_sentence)

        self.word_of_sent = []
        self.tokens = []
        punctuations = list(string.punctuation)
        punctuations.extend(["\'\'", "\"", "``", "--"])

        _index = []

        for i in range(len(tokenized_sents)):
            working_sentence = [word_tokenize(
                sentence) for sentence in tokenized_sents]

        # print("working_sentence = ", working_sentence)

        for i in range(len(working_sentence)):
            _sentenceWords = [
                word.lower() for word in working_sentence[i] if word not in punctuations]
            if len(_sentenceWords) > 2:
                self.word_of_sent.append(_sentenceWords)
            else:
                _index.append(i)

        # splitedSent contains sentences containing (num of words)> 2
        self.splitedSent = [tokenized_sents[i]
                            for i in range(len(tokenized_sents)) if i not in _index]
        # print("self.splitedSent = ", self.splitedSent)

        self.sentencesNum = len(self.splitedSent)

        for i in range(len(self.word_of_sent)):
            self.tokens.append([word for word in self.word_of_sent[i]
                                if word not in stopwords.words('english')])
            self.numOfWords += len(self.tokens[i])
        # print("self.word_of_sent = ", self.word_of_sent)

        stemmer = PorterStemmer()

        self.preprocTokens = []
        for j in range(len(self.tokens)):
            self.preprocTokens.append([stemmer.stem(words)
                                      for words in self.tokens[j]])

        _allTokens = itertools.chain.from_iterable(self.preprocTokens)

        self.distWordFreq = FreqDist(_allTokens)
        #print("self.distWordFreq = ", self.distWordFreq.r_Nr)

        self.preprocSentences = []
        for i in range(len(self.preprocTokens)):
            self.preprocSentences.append(' '.join(self.preprocTokens[i]))
        # print("self.preprocSentences = ", self.preprocSentences)

    def part_of_speech_tagging(self, list_of_sentences):
        tagged_sentences = []
        nlp = spacy.load('en_core_web_sm')
        for i in range(len(list_of_sentences)):
            tagged_sentences.append(list())
            doc = nlp(list_of_sentences[i])
            for token in doc:
                tagged_sentences[i].append((token.text, token.tag_))
        return tagged_sentences

    def preprocessing_titles(self, doc_titles):
        titles_list = doc_titles.splitlines()
        # titles_tagged = self.part_of_speech_tagging(titles_list) #Part-of-speech tagging of text sentences
        word_of_title = []
        tokens = []
        punctuations = list(string.punctuation)
        punctuations.extend(["\'\'", "\"", "``", "--"])

        for i in range(len(titles_list)):
            _titleWords = [word.lower() for word in word_tokenize(
                titles_list[i]) if word not in punctuations]
            if len(_titleWords) > 0:
                word_of_title.append(_titleWords)

        for i in range(len(word_of_title)):
            tokens.append([word for word in word_of_title[i]
                          if word not in stopwords.words('english')])

        stemmer = PorterStemmer()
        preproc_tokens = []
        for j in range(len(tokens)):
            if len(tokens[j]) > 0:
                preproc_tokens.append([stemmer.stem(words)
                                      for words in tokens[j]])
        preproc_titles = []

        for i in range(len(preproc_tokens)):
            preproc_titles.append(' '.join(preproc_tokens[i]))

        # print("preproc_titles = ", preproc_titles)
        return preproc_titles
