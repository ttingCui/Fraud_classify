# -*- coding = utf-8 -*-
# 2023/4/25 8:38

import os
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence

filedir = "../../cache/data_del_space"
with open("../../cache/get_label/textrank.txt", "w", encoding="UTF-8") as f1:
# with open("../../cache/get_label/test.txt", "w", encoding="UTF-8") as f1:
    for filename in os.listdir(filedir):
        path = os.path.join(filedir, filename)
        if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
            text_file = path
            # text = "手头紧找平安直通贷最快天放款月息最低DIGIT平台保证贷款成功送万个万里通积分详情点击URLdfgePLACE平安"
            text = codecs.open(text_file, "r", "utf-8").read()

            tr4w = TextRank4Keyword()
            tr4w.analyze(text=text, window=5, lower=True)

            f1.write("关键词：")
            for item in tr4w.get_keywords(num=20, word_min_len=1):
                f1.write(item.word+"\t"+str(item.weight))

            f1.write("关键短语：")
            f1.write(" ".join(tr4w.get_keyphrases(keywords_num=20, min_occur_num=2)))

            tr4s = TextRank4Sentence()
            tr4s.analyze(text=text, lower=True, source="all_filters")

            f1.write("摘要：")
            for item in tr4s.get_key_sentences(num=3):
                f1.write(str(item.index)+"\t"+str(item.weight)+"\t"+item.sentence)   # index是语句在文本中位置，weight是权重