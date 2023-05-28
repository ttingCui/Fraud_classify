# -*- coding = utf-8 -*-
# 2023/5/18 15:20
from zhkeybert import KeyBERT, extract_kws_zh
import jieba
import json
from tqdm import tqdm
lines = []
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
results = []
with open('../../cache/case_same/网黑案件.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()
    key_dic = {}
    for item in tqdm(test_data):
            content = item
            key_list = extract_kws_zh(content, kw_model, ngram_range=(1, 10))
            key_word = ''
            for i in key_list:
                key_word = key_word + i[0] + '、'
            key_word = key_word[:-1]
            key_dic[content] = key_word
            # item["案件关键词"] = key_word

with open('../../cache/keybert/test_with_key.txt','w')as f:
    for key, value in key_dic.items():
        f.write(key+"\t"+value+"\n")
    # json.dump(test_data, f, indent=2, ensure_ascii=False)