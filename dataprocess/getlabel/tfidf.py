# # -*- coding = utf-8 -*-
# # 2023/4/24 9:36
#
# import os
# import math
# import jieba
# import codecs
#
# documents = {}
# filedir = "../../cache/data_del_space"
# # 获取每个文件中的内容，存储在字典中
# for filename in os.listdir(filedir):
#     path = os.path.join(filedir, filename)
#     if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
#         text_file = path
#         text = codecs.open(text_file, "r", "utf-8").read()
#         documents[filename] = text
#
# # 分词并统计词频
# word_frequency = {}
# for doc in documents.values():
#     words = jieba.lcut(doc)  # 使用jieba进行分词
#     for word in words:
#         if word not in word_frequency:
#             word_frequency[word] = 0
#         word_frequency[word] += 1
#
# # 计算idf值
# idf = {}
# total_documents = len(documents)
# for word in word_frequency:
#     documents_with_word = sum(word in doc for doc in documents.values())
#     idf[word] = math.log(total_documents / (1 + documents_with_word))
#
# # 计算tf-idf值
# tfidf = {}
# for file_name, doc in documents.items():
#     tfidf[file_name] = {}
#     words = jieba.lcut(doc)
#     for word in words:
#         tf = words.count(word) / len(words)  # 计算词频
#         tfidf[file_name][word] = tf * idf[word]
#
# # 打印结果到文件中
# with open("../../cache/get_label/tfidf.txt", "w", encoding="UTF-8") as f1:
#     for file_name, file_tfidf in tfidf.items():
#         f1.write("文件:"+file_name+"\n")
#         file_tfidf = dict(sorted(file_tfidf.items(), key=lambda x: x[1]))
#         for word, score in file_tfidf.items():
#             f1.write(word+":"+str(score)+"\n")


# import math
# import os
#
# def calculate_tfidf(corpus):
#     # 构建词频字典
#     word_frequency = {}
#     for doc in corpus:
#         for char in doc:
#             if char not in word_frequency:
#                 word_frequency[char] = 0
#             word_frequency[char] += 1
#
#     # 计算idf值
#     idf = {}
#     total_documents = len(corpus)
#     for char in word_frequency:
#         documents_with_char = sum(char in doc for doc in corpus)
#         idf[char] = math.log(total_documents / (1 + documents_with_char))
#
#     # 计算tf-idf值
#     tfidf = {}
#     for doc in corpus:
#         tfidf[doc] = {}
#         doc_length = len(doc)
#         for char in doc:
#             tf = doc.count(char) / doc_length  # 计算词频
#             tfidf[doc][char] = tf * idf[char]
#
#     return tfidf, word_frequency
#
#
# filedir = "../../cache/data_del_space"
# tfidfdir = "../../cache/get_label/tfidf"
#
# for filename in os.listdir(filedir):
#     path = os.path.join(filedir, filename)
#     if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
#         text_file = path
#         # 读取文件并将每行数据存储在列表中
#         corpus = []
#         with open(text_file, 'r', encoding='utf-8') as file:
#             for line in file:
#                 corpus.append(line.strip())
#
#         # 计算tf-idf
#         result, word_frequency = calculate_tfidf(corpus)
#
#         # 打印结果
#         tfidfpath = os.path.join(tfidfdir, filename)
#         with open(tfidfpath, "w", encoding="UTF-8") as f1:
#             for doc, doc_tfidf in result.items():
#                 f1.write("文档:"+doc+"\n")
#                 # print("文档:", doc)
#                 doc_tfidf = dict(sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True))
#                 for char, score in doc_tfidf.items():
#                     # print(char, ":", score)
#                     f1.write(char + ":" + str(score) + "\n")
#                     # print(char, ":", score)
#             # 打印词频文件
#             word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))
#             f1.write("词频:" + "\n")
#             for word, value in word_frequency.items():
#                 f1.write(word + ":" + str(value) + "\n")


#encoding: utf-8

import sys

from math import log

''' usage:
	python tfidftrain.py train_data.txt model.txt
'''


# 清理列表中的空值，去除空字符串
def cleanlist(lin):

	rs = []
	for lu in lin:
		if lu:
			rs.append(lu)

	return rs


#  计算每个特征的idf值
def calculateidf(featc, ndata):

	rsd = {}
	_nd = float(ndata)
	for k, v in featc.items():
		rsd[k] = log(ndata / float(v))

	return rsd


# 计算每个特征的tf值
def calculatetf(featfd, classd):

	rsd = {}
	for clas, fd in featfd.items():
		_nfeatc = float(classd[clas])
		tmp = {}
		for ft, freq in fd.items():
			tmp[ft] = float(freq) / _nfeatc
		rsd[clas] = tmp

	return rsd


# 将tf和idf值合并成tfidf值
def buildtfidf(tfd, idfd, build_unk=True, avg_unkv=False, min_unkv=False, unk_percent=0.001, unk_max=16):

	rsd = {}
	for clas, fd in tfd.items():
		tmp = {}
		if build_unk:
			if avg_unkv:
				tfidfl = []
			elif min_unkv:
				m_unkv = 0.0
		for ft, tf in fd.items():
			_tfidf = tf * idfd[ft]
			tmp[ft] = _tfidf
			if build_unk:
				if avg_unkv:
					tfidfl.append(_tfidf)
				elif min_unkv and _tfidf < m_unkv:
					m_unkv = _tfidf
		if build_unk:
			if avg_unkv:
				ncount = max(min(len(tfidfl) * unk_percent, unk_max), 1)
				tfidfl.sort()
				unk_tfidf = 0.0
				for _tfidf in tfidfl[:ncount]:
					unk_tfidf += _tfidf
				tmp["<unk>"] = unk_tfidf / ncount
			elif min_unkv:
				tmp["<unk>"] = m_unkv
			else:
				tmp["<unk>"] = 0.0
		rsd[clas] = tmp

	return rsd


# 对给定的字典进行归一化处理，将频率值除以总数得到概率分布
def norm(cld, nd):

	rsd = {}
	_nd = float(nd)
	for k, v in cld.items():
		rsd[k] = float(v) / _nd

	return rsd

# srcf: source file to train the model with format: _class_label _feature1 _feature2 ... _featuren
# modelf: file to save the trained model
# norm_feat: if a feature repeated several times in a line, reduce its frequency to 1 in that line
# norm_bias: normalize bias to a probability distribution

# 主要处理逻辑函数，包括读取训练数据、计算TF和IDF值，合并成TF-IDF值，并保存模型
def handle(srcf, modelf, norm_feat=False, norm_bias=False):

	feat_freq = {}
	class_freq = {}

	ndata = 0
	nfeat = 0
	feat_n = {}
	class_n = {}

	# count
	with open(srcf, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = cleanlist(tmp.decode("utf-8").split())
				_clas, _features = tmp[0], tmp[1:]
				if norm_feat:
					_features = list(set(_features))
				for _f in _features:
					if _clas in feat_freq:
						if _f in feat_freq[_clas]:
							feat_freq[_clas][_f] += 1
						else:
							feat_freq[_clas][_f] = 1
					else:
						feat_freq[_clas] = {_f:1}
					feat_n[_f] = feat_n.get(_f, 0) + 1
				class_freq[_clas] = class_freq.get(_clas, 0) + 1
				class_n[_clas] = class_n.get(_clas, 0) + len(_features)
				nfeat += len(_features)
				ndata += 1

	# calculate IDF
	feat_n = calculateidf(feat_n, nfeat)
	# calculate TF
	feat_freq = calculatetf(feat_freq, class_n)

	# merge TFIDF
	model = buildtfidf(feat_freq, feat_n)

	if norm_bias:
		class_n = norm(class_freq, ndata)

	ens = "\n".encode("utf-8")
	with open(modelf, "w", encoding="UTF-8") as f:
		for key, value in model.items():
			sorted_dict = dict(sorted(value.items(), key=lambda item: item[1], reverse=True))
			f.write(key)
			f.write(":\n")
			for sub_key, sub_value in sorted_dict.items():
				f.write(sub_key)
				f.write(": ")
				f.write(str(sub_value))
				f.write("\n")
			f.write("\n")
		# f.write(repr(model).encode("utf-8"))
		# f.write(ens)
		# f.write(repr(class_n).encode("utf-8"))
		# f.write(ens)

if __name__ == "__main__":
	# handle(sys.argv[1], sys.argv[2])
	handle("../../cache/merged_data.txt", "../../cache/get_label/tfidf_model.txt")


# if __name__ == "__main__":
#     data_directory = "path/to/your/data/directory"
#     model_file = "model.txt"
#
#     classes = [
#         "AD_Loan",
#         "AD_Network_service",
#         "AD_Other",
#         "AD_Real_estate",
#         "AD_Retail",
#         "FR_Financial",
#         "FR_Other",
#         "FR_Phishing(Bank)",
#         "FR_Phishing(Other)",
#         "IL_Escort_service",
#         "IL_Fake_ID_and_invoice",
#         "IL_Gambling",
#         "IL_Political_propaganda",
#         "No_fraud"
#     ]
#
#     for class_name in classes:
#         files_path = f"{data_directory}/{class_name}/*.txt"
#         files = glob.glob(files_path)
#         for file in files:
#             handle(file, model_file)
