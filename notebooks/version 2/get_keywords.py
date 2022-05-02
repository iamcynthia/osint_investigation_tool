import math
import pickle
import pandas as pd
import numpy as np
import re
import os
from LAC import LAC
from zhon.hanzi import punctuation as zhPunc
from string import punctuation as enPunc
from collections import Counter

main_dir = '/home/cynthiachan'
file_dir = 'Corpus/for_text_classification'

STOPWORD_PATH = f'{main_dir}/Corpus/baidu_stopwords.txt'
__stopwords = list("\xa0\r\n\t\f\v□ ") + list(enPunc) + list(zhPunc)
# __converter = OpenCC("tw2sp")

with open(STOPWORD_PATH, "r") as f:
    lines = f.read().splitlines()
    __stopwords.extend(lines)
    for line in lines:
        __stopwords.append(line)#__converter.convert(line))


def __get_tfidfs(counters, top):
    """取得各篇文章的關鍵字及其透過TF-IDF加權分數的前20名
    
    :param counters: 各篇文章詞頻
    :param top: 前?名
    :return 各篇文章的前?名關鍵詞(dataframe)
    """
    def tf(word, counter):
        return counter[word] / sum(counter.values())

    def idf(word, counters):
        return math.log2(len(counters) / sum(1 for counter in counters if word in counter))

    data = []
    for idx, counter in enumerate(counters):
        data.extend([idx, word, round(score, 5)] for word, score in sorted([
            (word, tf(word, counter) * idf(word, counters)) for word in counter
        ], key=lambda x: x[1], reverse=True)[:top])
    
    return pd.DataFrame(data, columns=["idx", "word", "score"])

#classes = os.listdir(f'{main_dir}/{file_dir}/word_segments')
classes = ['财经', '教育', '社会', '时政', '科技', '游戏', '体育', '时尚', '娱乐', '家居', '彩票', '房产', '星座', '股票']
for cls in classes:
    with open(f'{main_dir}/{file_dir}/word_segments/{cls}.pickle', 'rb') as f:
        loaded_data = pickle.load(f)
    print('Data Type: ', cls)
    print('Loaded data length: ',len(loaded_data))
    
    data = []
    for lists in loaded_data:
        temp = []
        for words in lists:
            temp.append(words.replace('\xa0', ''))
        data.append(temp)
    print('Used data length: ', len(data))
    all_doc_cnt = [Counter(eachlist) for eachlist in data] # 計算詞頻
    tfidf = list(__get_tfidfs(all_doc_cnt, 10)['word'])
    with open(f'{main_dir}/{file_dir}/topic_words/{cls}.pickle', 'wb') as f:
        pickle.dump(tfidf, f)
    print('Finished extract topic words from %s'%cls)
    print('----------------------------------------')
