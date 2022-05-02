#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import os
import random
import re
import time
from collections import Counter
from contextlib import closing
from string import punctuation as enPunc

import gensim
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec
from LAC import LAC
from opencc import OpenCC
from zhon.hanzi import punctuation as zhPunc


# In[4]:


## 設定檔案位址
user_name = '/Users/cynthiachan/Downloads'
file_dir = '/公司資料/OSINT'


# ## 1 Target Page : V 3.0 (Api version)

# - input: 某taskId包含之網址
# - output: 網頁重新排序後之列表
# ---------------------
# 0. (載入)預訓練詞向量
# 1. 爬蟲&資料預處理：爬網頁文章內容、分詞並計算tfidf
# 2. 排序模組：匯入詞向量並進行增量訓練，取得搜尋詞詞向量，取得文件向量並與搜尋詞比較相似性

# ### 1.1 爬蟲&資料預處理

# #### 1.1.1 爬蟲取得網頁內容文章

# In[2]:


def __get_content_texts(google_search_rs):
    """爬蟲取得網頁內容文章
    
    :param google_search_rs: 特定taskID中所有url
    :return context: 網頁文章，以句子分開
    """
    articleLength, context = [], []
    for idx, url in enumerate(google_search_rs):
        print(idx +1, url)
        try:
            with closing(requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36"
            }, timeout=30, verify=False)) as resp:
                if resp.ok and resp.headers.get("Content-Type", "").startswith("text/html"):
                    soup = BeautifulSoup(resp.content.decode("utf-8", "ignore"), "lxml")
                    if soup.body:
                        for tree in soup.find_all([
                            "a", "audio", "button", "head", "iframe", "img",
                            "input", "link", "meta", "noscript", "object", "script",
                            "select", "source", "style", "textarea", "title", "video"
                        ]):
                            tree.extract()
                        if soup.body.find("main"):
                            texts = [
                                re.sub(r"[\xa0\r\n\t\f\v ]+", " ", text.strip())
                                for text in soup.body.find("main").get_text().splitlines()
                                if re.search(r"\w", text)
                            ]
                        else:
                            texts = [
                                re.sub(r"[\xa0\r\n\t\f\v ]+", " ", text.strip())
                                for text in soup.body.get_text().splitlines()
                                if re.search(r"\w", text)
                            ]
            context.append(texts)
        except Exception as e:
            context.append([""])
            print(e)
        
        length = sum(len(s) for s in texts)          
        articleLength.append(length if length != 0 else 0)
    
    return context, articleLength


# #### 1.1.2 分詞&計算詞頻

# In[13]:


STOPWORD_PATH = f'{user_name}{file_dir}/stopwords-master/baidu_stopwords.txt'
__stopwords = list("\xa0\r\n\t\f\v ") + list(enPunc) + list(zhPunc)
__converter = OpenCC("tw2sp")

with open(STOPWORD_PATH, "r") as f:
    lines = f.read().splitlines()
    __stopwords.extend(lines)
    for line in lines:
        __stopwords.append(__converter.convert(line))


# In[14]:


__valid_tags = ["n", "nz", "v", "vd", "vn", "a", "ad", "an", "d", "PER", "LOC", "ORG", "TIME"]
__lac = LAC(mode="lac")
__lac.load_customization(f'{user_name}{file_dir}/data/custom_dict.txt')


# In[15]:


def __get_named_entities(texts):
    """取得文章命名實體
    
    :param texts: 所有網頁文章(list)
    :return for_target_page(目標頁所需之命名實體), for_info_extract(關鍵資訊擷取所需之命名實體)
    """
    for_target_page, for_info_extract = [], []
    for idx, text in enumerate(texts):
        try:
            result: list[(str, str)] = []
            if isinstance(text, (str, list)) and len(text):
                try:
                    lac_result = __lac.run(text if isinstance(text, list) else [text])
                except:
                    lac_result = __lac.run([" ".join(text)]) if isinstance(text, list) else []
                finally:
                    for words, tags in lac_result:
                        result.extend((words[idx], tags[idx]) for idx in range(len(words)) if len(words[idx])>1)
#             print(result) # [('中國國防科技', 'ORG'), ('信息', 'n'), ('中心', 'n'),....

            _for_target_page = [
                # 篩選出屬於合法詞性類別標籤且不包含在斷詞中的分詞
                word for word, tag in result
                if tag in __valid_tags
                and word not in __stopwords
                and not re.search(r"[0-9]", word)
            ]
            for_target_page.append(_for_target_page)

            _for_info_extract = [
                # 篩選出關鍵詞彙(客製化需求欄位，未來來源可能會調整)
                word for word, tag in result
                if tag in ["ORG", "PER", "nz", "an", "vn"]
                and word not in __stopwords
                and not re.search(r"[0-9]", word)
            ]
            for_info_extract.append(_for_info_extract)
            
        except Exception as e:
            for_target_page.append([])
            for_info_extract.append([])
    return for_target_page, for_info_extract


# #### 1.1.3 計算tfidf

# In[12]:


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


# ### 1.2 排序模組

# #### 1.2.1 匯入詞向量並進行增量訓練

# In[11]:


MODEL_PATH = f'{user_name}{file_dir}/model/api_version/word2vec.model'

def __get_trained_model(query, corpus):
    """詞向量增量訓練

    :param query(str): 搜尋詞
    :param corpus: 語料庫，預期帶入文章分詞結果集合
    :return: 增量訓練後的模型
    """
    if os.path.isfile(MODEL_PATH): #該路徑是否為文件
        model = Word2Vec.load(MODEL_PATH)
    else:
        return None

    words, tags = __lac.run(query)

    model.build_vocab(corpus, update=True, trim_rule=lambda word, count, min_count:
                      gensim.utils.RULE_KEEP if word in words else gensim.utils.RULE_DEFAULT)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_PATH)

    return model


# #### 1.2.2 取得搜尋詞詞向量

# In[10]:


def __get_vector(query, model):
    """取得搜尋詞及其向量

    若詞向量中有此搜尋詞，則取得該詞彙之向量，
    若詞向量中無此搜尋詞，則進行分詞，再尋找該詞彙之向量

    :param query(str): 搜尋詞
    :param model(Word2Vec): 詞向量的相關模型
    :return: vector
    """
    try:
        vector = model.wv.vectors[model.wv.key_to_index[query]]
    except:
        words, tags = __lac.run(query)
        vectors = [
            model.wv.vectors[model.wv.key_to_index[word]]
            for word in words if word not in __stopwords
        ]
        vector = sum(vectors) / len(vectors)
    return vector


# #### 1.2.3 取得文件向量並與搜尋詞比較相似性

# In[9]:


def __get_similarity(google_search_rs, top, query):
    contexts, articleLength = __get_content_texts(google_search_rs)
    for_target_page, for_info_extract = __get_named_entities(contexts)
    all_doc_count = [Counter(for_target_page[i]) for i in range(len(for_target_page))] #取得各篇文章的term frequency
    tfidfs = __get_tfidfs(all_doc_count,top) #所有網頁內容分詞的tfidfs，return dataframe
    model = __get_trained_model(query, contexts) #詞向量模型
    query_vector = __get_vector(query, model) #搜尋詞的詞向量
    
    vector = np.zeros(250)
    distance, words = [], []
    for idx in range(len(google_search_rs)):
        for row in tfidfs.loc[tfidfs["idx"] == idx].itertuples():
            try:
                vector += model.wv.vectors[model.wv.key_to_index[row.word]] * row.score
            except:
                continue
            else:
                words.append(row.word)
        else:
            total = len(words)
            distance.append(math.dist(vector if 0 == total else (vector / total), query_vector)) #各篇文章與搜尋詞的相似度

    for sort, idx in enumerate(
        sorted(range(len(google_search_rs)), key=lambda idx: distance[idx] 
               if 400 < articleLength[idx] else 100, reverse=False)):
        print("Article %d: %s"%(sort+1,google_search_rs[idx]))


# In[ ]:





# ## 2 Info Extraction: V 3.0 (Api version)

# In[3]:


def __get_keywords(counts, top):
    keyword_count = [Counter(for_info_extract[i]) for i in range(len(for_info_extract))] #取得各篇文章的term frequency (specific named entities)
    keyword_tfidfs = __get_tfidfs(keyword_count,top)
    return keyword_tfidfs










