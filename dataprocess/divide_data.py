# -*- coding = utf-8 -*-
# 2023/4/4 15:33

import os
from matplotlib import pyplot as plt
import random
plt.rcParams['font.family'] = 'SimHei'

# 获取各个的类别数量
def get_class_num(filedir):
    lenout = {}
    for filename in os.listdir(filedir):
        path = os.path.join(filedir, filename)
        # 判断是否是文件还是目录需要用绝对路径
        if os.path.isfile(path):
            with open(path, "r", encoding="UTF-8") as f:
                lenout[filename] = len(f.readlines())
    return lenout


# 获取类别数量图象
def get_img(len_dic, title):
    fig, ax = plt.subplots()

    filename = list(len_dic.keys())
    length = list(len_dic.values())
    ax.bar(filename, length)

    for a, b in zip(filename, length):  # 柱子上的数字显示
        plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=7)

    plt.xlabel("class", fontsize=8)
    plt.xticks(rotation=90, fontsize=8)
    ax.set_ylabel(u'数量')
    ax.set_title(title)

    plt.show()


# 每个文件等量抽取一定数量的条数
def get_num_file(fileindir, fileoutdir, num=112):
    for filename in os.listdir(fileindir):
        path_in = os.path.join(fileindir, filename)
        path_out = os.path.join(fileoutdir, filename)
        if os.path.isfile(path_in):  # 判断是否是文件还是目录需要用绝对路径
            with open(path_in, "r", encoding="UTF-8") as f1:
                lines = f1.readlines()
                random.shuffle(lines)  # 按行打乱顺序
                with open(path_out, "w", encoding="UTF-8") as f2:
                    i = 0
                    for line in lines:
                        # if line.strip() or line == '\n':  # 检查非空行
                        f2.write(line)
                        i += 1
                        if i == num:  # 达到指定数量后退出循环
                            break


# 抽取为训练集、验证机和测试集
def split_num(fileindir, filetraindir, filedevdir, filetestdir, dev_num=40, test_num=40):
    for filename in os.listdir(fileindir):
        path_in = os.path.join(fileindir, filename)
        path_out_train = os.path.join(filetraindir, filename)
        path_out_dev = os.path.join(filedevdir, filename)
        path_out_test = os.path.join(filetestdir, filename)
        if os.path.isfile(path_in):  # 判断是否是文件还是目录需要用绝对路径
            with open(path_in, "r", encoding="UTF-8") as f1:
                lines = f1.readlines()
                random.shuffle(lines)  # 按行打乱顺序
                with open(path_out_dev, "w", encoding="UTF-8") as f2, open(path_out_test, "w", encoding="UTF-8") as f3, open(path_out_train, "w", encoding="UTF-8") as f4:
                    # 写入文件
                    for i, line in enumerate(lines):
                        if i < dev_num:
                            f2.write(line)
                        elif i < dev_num + test_num:
                            f3.write(line)
                        else:
                            f4.write(line)


# 合并文件夹下所有文件数据到一个文件中
def merge_files(filedir, fileout):
    with open(fileout, "w", encoding="UTF-8") as f1:
        for filename in os.listdir(filedir):
            path = os.path.join(filedir, filename)
            if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
                with open(path, "r", encoding="UTF-8") as f2:
                    lines = f2.readlines()
                    f1.write("".join(lines))


# 打乱某文件的数据
def random_file(filein, fileout):
    with open(filein, "r", encoding="UTF-8") as f1:
        lines = f1.readlines()
        random.shuffle(lines)  # 按行打乱顺序
        with open(fileout, "w", encoding="UTF-8") as f2:
            # 写入文件
            f2.write("".join(lines))


if __name__ == '__main__':
    # 获取原始数据不同类别信息
    # get_img(get_class_num("../cache/data_class"))
    # 每个类别抽取112条数据
    # get_num_file("../cache/data_class", "../cache/data_same", num=112)
    # get_img(get_class_num("../cache/data_same"))
    # 划分train dev test数据
    # split_num("../cache/data_same", "../cache/data_train", "../cache/data_dev", "../cache/data_test")
    # get_img(get_class_num("../cache/data_train"))
    # get_img(get_class_num("../cache/data_dev"))
    # get_img(get_class_num("../cache/data_test"))
    # 合并dev test数据并打乱存储
    # merge_files("../cache/data_dev", "../message/new_data/dev.txt")
    # merge_files("../cache/data_test", "../message/new_data/test.txt")
    # random_file("../message/new_data/dev.txt", "../message/new_data/dev.txt")
    # random_file("../message/new_data/test.txt", "../message/new_data/test.txt")
    # 抽取训练集数据作为小样本数据集
    # get_num_file("../cache/data_train", "../cache/data_fewshot_train/fewshot4", num=4)
    # merge_files("../cache/data_fewshot_train/fewshot4", "../message/new_data/fewshot_4/train.txt")
    # random_file("../message/new_data/fewshot_4/train.txt", "../message/new_data/fewshot_4/train.txt")

    # get_num_file("../cache/data_train", "../cache/data_fewshot_train/fewshot8", num=8)
    # merge_files("../cache/data_fewshot_train/fewshot8", "../message/new_data/fewshot_8/train.txt")
    # random_file("../message/new_data/fewshot_8/train.txt", "../message/new_data/fewshot_8/train.txt")
    #
    # get_num_file("../cache/data_train", "../cache/data_fewshot_train/fewshot16", num=16)
    # merge_files("../cache/data_fewshot_train/fewshot16", "../message/new_data/fewshot_16/train.txt")
    # random_file("../message/new_data/fewshot_16/train.txt", "../message/new_data/fewshot_16/train.txt")
    #
    # get_num_file("../cache/data_train", "../cache/data_fewshot_train/fewshot32", num=32)
    # merge_files("../cache/data_fewshot_train/fewshot32", "../message/new_data/fewshot_32/train.txt")
    # random_file("../message/new_data/fewshot_32/train.txt", "../message/new_data/fewshot_32/train.txt")

    # 合并停用词表
    # merge_files("../cache/stopwords", "../cache/stopwords/stopwords.txt")
    # 获取原始数据不同类别信息
    # get_img(get_class_num("../cache/case"))
    # 每个类别抽取874条数据
    # get_num_file("../cache/case", "../cache/case_same", num=875)
    # get_img(get_class_num("../cache/case_same"))
    # 划分train dev test数据
    # split_num("../cache/case_class", "../cache/case_train", "../cache/case_dev", "../cache/case_test", dev_num=405, test_num=405)
    # get_img(get_class_num("../cache/case_train"))
    # get_img(get_class_num("../cache/case_dev"))
    # get_img(get_class_num("../cache/case_test"))
    # 合并dev test数据并打乱存储
    # merge_files("../cache/case_dev", "../case/dev.txt")
    # merge_files("../cache/case_test", "../case/test.txt")
    # random_file("../case/dev.txt", "../case/dev.txt")
    # random_file("../case/test.txt", "../case/test.txt")
    # 抽取训练集数据作为小样本数据集
    # get_num_file("../cache/case_train", "../cache/case_fewshot_train/fewshot4", num=4)
    # merge_files("../cache/case_fewshot_train/fewshot4", "../case/fewshot_4/train.txt")
    # random_file("../case/fewshot_4/train.txt", "../case/fewshot_4/train.txt")
    # #
    # get_num_file("../cache/case_train", "../cache/case_fewshot_train/fewshot8", num=8)
    # merge_files("../cache/case_fewshot_train/fewshot8", "../case/fewshot_8/train.txt")
    # random_file("../case/fewshot_8/train.txt", "../case/fewshot_8/train.txt")
    # #
    # get_num_file("../cache/case_train", "../cache/case_fewshot_train/fewshot16", num=16)
    # merge_files("../cache/case_fewshot_train/fewshot16", "../case/fewshot_16/train.txt")
    # random_file("../case/fewshot_16/train.txt", "../case/fewshot_16/train.txt")
    # #
    # get_num_file("../cache/case_train", "../cache/case_fewshot_train/fewshot32", num=8)
    # merge_files("../cache/case_fewshot_train/fewshot32", "../case/fewshot_32/train.txt")
    # random_file("../case/fewshot_32/train.txt", "../case/fewshot_32/train.txt")
    # get_img(get_class_num("../cache/data_del_space"))
    get_img(get_class_num("../cache/case"), title=u"训练集")
    get_img(get_class_num("../cache/case_test_class"), title=u"测试集")