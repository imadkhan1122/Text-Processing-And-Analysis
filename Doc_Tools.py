from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import re
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt


class DocTools():

    # Category 1: Provided Methods
    def __init__(self, pth):
        self.pth = pth
        self.punctuation = ['.', '?', '!', ',', ';', ':']
        self.common_words = ["as", "i", "am", "his", "that", "he", "was", "for", "on", "are", "with", "they", "be", "at",
                             "one", "have", "this", "from",
                             "by", "hot", "word", "but", "what", "some", "is", "it", "you", "or", "had", "the", "of",
                             "to", "and", "a", "in", "we"]
    def load_document(self):
        
        """
        Read a file and create list of paragraphs
        :param file_name: name of file in the project (or full path to the file)
        """
        reader = open(self.pth, 'r')
        return [line.rstrip() for line in reader]

    def word_count(self):
        
        """
        Get current documents word count
        :return: number of words in essay
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        words = str_.split()
        
        return len(words)

    def character_count(self):
        
        """
        Get current documents character count (include all characters except new line characters)
        :return: number of characters in essay
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        
        return len(str_)

    def sentence_count(self):
        
        """
        Get current documents sentence count
        :return: number of sentences in essay
        """
        lst = self.load_document()
        
        str_ = ','.join(lst).strip()
        sents = re.split('[?|.|!]', str_)
        sents_lst = [x for x in sents if x]
        # sents = sent_tokenize(str_)
        
        return len(sents_lst)

    #Category 3: Essay Analysis
    def longest_word(self):
        
        """
        Get the longest word used in the document and length.
        Return a list of words with the same length
        :return: ([words], length)
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        stop = set(self.common_words + self.punctuation)
        fil_lst = [i for i in word_tokenize(str_.lower()) if i not in stop]
        longest = max(fil_lst, key=len)
        same_len = [i for i in fil_lst if len(i)==len(longest)]

        return same_len, len(longest)

    def average_word_length(self):
        
        """
        Get the average length of words used in the document
        :return: average length rounded to 2 decimals
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        w = [len(word) for word in str_.rstrip().split(" ")]
        w_avg = sum(w)/len(w)
        
        return round(w_avg, 2)

    def average_sentence_length(self):
        
        """
        Get the average length of sentences used in the document (measured in words)
        :return: average length rounded to 2 decimals
        """
        lst = self.load_document()
        
        str_ = ','.join(lst).strip()
        sents = re.split('[?|.|!]', str_)
        sents_lst = [x for x in sents if x]
        avg_len = sum(len(x.split()) for x in sents_lst) / len(sents_lst)
        
        return round(avg_len, 2)

    def longest_sentence(self):
        
        """
        Get the longest sentence and lenth of sentence in the document
        Return a list of sentneces with the same length
        :return: ([sentences], length)
        """
        
        lst = self.load_document()
        
        str_ = ','.join(lst).strip()
        sents = re.split('[?|.|!]', str_)
        sents_lst = [x for x in sents if x]
        long_sent = max(sents_lst, key=len)
        same_len = [i for i in sents_lst if len(i.split())==len(long_sent.split())]
        
        return same_len, len(long_sent.split())

    def shortest_sentence(self):
        
        """
        Get the shortest sentence and lenth of sentence in the document
        Return a list of sentneces with the same length
        :return: ([sentences], length)
        """
        lst = self.load_document()
        str_ = ','.join(lst).strip()
        sents = re.split('[?|.|!]', str_)
        sents_lst = [x for x in sents if x]
        short_sent = min(sents_lst, key=len)
        same_len = [i for i in sents_lst if len(i.split())==len(short_sent.split())]
        
        return same_len, len(short_sent.split())

    def most_frequent_words(self, n):
        
        """
        Get the n most frequent words from the document (ignore capitilization and punctuiation).
        Ignore all words in the ignore_list
        You may break ties in any way.
        :param n: Number of words to return
        :return: List of most frequent words in order
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        stop = set(self.common_words + self.punctuation)
        fil_lst = [i for i in word_tokenize(str_.lower()) if i not in stop]
        counter = Counter(fil_lst)
        # most_common() produces k frequently encountered
        # input values and their respective counts.
        most_occur = counter.most_common(n)

        return [i[0] for i in most_occur]

    def num_distinct_words(self):
        
        """
        Get the number of distinct words (ignore punctuation and capitalization)
        :return: Number of distinct words
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        stop = set(self.common_words + self.punctuation)
        fil_lst = [i for i in word_tokenize(str_) if i not in stop]
        unique = set(fil_lst)
        
        return len(unique)

    #Category 5: Advanced Analysis
    def words_by_prefix(self, prefix):
        
        """
        Get a list of all words in the document with the prefix provided. Sort alphabetical. Ignore capitalization
        :param prefix:
        :return:
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        stop = set(self.common_words + self.punctuation)
        fil_lst = [i for i in word_tokenize(str_.lower()) if i not in stop]
        prefix_similar = set(word for word in fil_lst if word.startswith(prefix))
        
        return list(prefix_similar)

    def character_fingerprint(self):
        
        """
        Generate a character fingerprint dictionary of letters (ignoring capitalization)
        Do not include any non-letter characters (only a-z)
        show plot using matplotlib and include in your README
        :return: dictionary finger_print of letters with their counts
        """
        lst = self.load_document()
        str_ = ','.join(lst)
        str_ = [i for i in str_ if i not in self.punctuation]
        res = {}
        for key in str_:
            keys = key.lower()
            if keys.isalpha():
                res[keys] = res.get(keys, 0) + 1
        
        return res
    
    def letters_barChart(self):
        
        lst = self.load_document()
        str_ = ','.join(lst)
        str_ = [i for i in str_ if i not in self.punctuation]
        total = len(str_)
        eng_alphs = self.character_fingerprint()
        sort_keys = eng_alphs.items()
        new_items = sorted(sort_keys)
        Letters = [l[0] for l in new_items]
        Frequency = [round((l[1]*100)/total, 3) for l in new_items]
        plt.figure(figsize = (10, 5))
        # creating the bar plot
        plt.bar(Letters, Frequency, color ='maroon',
            width = 0.4)
        plt.xlabel("Letters from a-z")
        plt.ylabel("Frequency in %")
        plt.title("Letters Frequency Bar Chart")
        plt.savefig('plot.png')
        return str('Chart Created by name: plot.png')

    def auto_complete(self, input_string):
        
        """
        Return the top 3(at most) results as a list by current usage in the document where the input is a prefix
        Ignore capitalization.
        You may break times in any way you want.
        If there are less than 3 words with the input_string as prefix return fewer results
        :param input_string:
        :return: list of recommendations
        """
        lst = self.load_document()
        
        str_ = ','.join(lst)
        stop = set(self.common_words + self.punctuation)
        fil_lst = [i for i in word_tokenize(str_.lower()) if i not in stop]
        counter = Counter(fil_lst)
        # most_common() produces k frequently encountered
        # input values and their respective counts.
        most_occur = counter.most_common()
        auto_lst = [i[0] for i in most_occur if i[0].startswith(input_string)]
        return input_string, auto_lst[:3]


    #Category 5: Tools
    def find_word(self, word):
        
        """
        Find all the positions of a word (like 5th word in essay should return 5 NOT 4) and the sentence it is a part of.
        (ignore punctuation and capitalization)
        :param word: word to search for
        :return: List of positions and list of sentences
        """
        
        lst = self.load_document()
        str_ = ','.join(lst)
        str_lst = str_.split(' ')
        indices = [i+1 for i, x in enumerate(str_lst) if x.lower() == word.lower()]
        
        return indices

    def replace_word(self, original, new):
        
        """
        Replace all instances of one word with new word. Maintain punctuation but do not ignore capitalization
        :param original: word to replace
        :param new: word to replace with
        :return: number of changes, list of paragraphs with changes
        """   
        new_lst = []
        count = 0
        lst = self.load_document() 
        for l in lst:
            txt = []
            str_ = l.split()
            for s in str_:
                s = [i for i in s if i not in self.punctuation]
                s = ''.join(s)
                if s.lower() == original.lower():
                    count+=1
                    if s.istitle():
                        txt.append(new.title())
                    else:
                        txt.append(new.lower())     
                else:
                    txt.append(s)
            new_lst.append(' '.join(txt))
            
        return count, new_lst

    def spell_check(self, dictionary_pth):
        
        """
        Use dictionary.txt to check for correct spelling of words. Ignore punctuation and capitalization.
        :return: (number_wrong, [indices])
        """
        words = []
        with open(dictionary_pth, 'r') as f:
            for line in f:
                lin = line.strip('\n')
                words.append(lin)
        
        lst = self.load_document()
        str_ = ','.join(lst)
        str_ = [i for i in str_ if i not in self.punctuation]
        filtered_str_ = ''.join(str_)
        str_lst = [i for i in filtered_str_.split() if i.islower()==True]
        
        count = 0
        ind = []
        for e, w in enumerate(str_lst):
            if w.lower() not in self.common_words:
                # print(w)
                word = ''
                n = 0
                if w not in words:
                    for W in words:
                        ratio = fuzz.ratio(w, W)
                        if ratio > n:
                            n = ratio
                            word = W
                    if n < 100:
                        count+=1
                        ind.append(e)
                
        return count, ind

    def write_document(self, file_name, original, new):
        
        """
        Write the current document to a new file
        :param file_name:
        :return:
        """        
        with open(file_name, 'w') as f:
            count, lst = self.replace_word(original, new)
            for l in lst:
                f.write(l+'\n')
        return str('File Created')

    
