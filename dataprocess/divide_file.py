# -*- coding = utf-8 -*-
# 2023/3/8 22:58
import os, random

# 合并所有数据到一个文件
def merge_files(filedir, fileout):
    with open(fileout, "w", encoding="UTF-8") as f1:
        for filename in os.listdir(filedir):
            path = os.path.join(filedir, filename)
            if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
                with open(path, "r", encoding="UTF-8") as f2:
                    lines = f2.readlines()
                    f1.write("".join(lines))


# 先分为训练集和测试集，再从训练集分出验证集
def split_ratio(fileall, filefir, fileother, fir_ratio, other_ratio):
    with open(fileall, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        n_total = len(lines)  # 获取数据集的总长度

        fir_offset = int(n_total * fir_ratio)
        other_offset = int(n_total * (fir_ratio + other_ratio))
        random.shuffle(lines)  # 按行打乱顺序
        fir_data = open(filefir, 'w', encoding="UTF-8")
        other_data = open(fileother, 'w', encoding="UTF-8")

        # 写入文件
        for i, line in enumerate(lines):
            if i < fir_offset:
                fir_data.write(line)
            elif i < other_offset:
                other_data.write(line)

        fir_data.close()
        other_data.close()


def split_num(fileall, filefir, fileother, other_num):
    with open(fileall, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        n_total = len(lines)  # 获取数据集的总长度

        random.shuffle(lines)  # 按行打乱顺序
        fir_data = open(filefir, 'w', encoding="UTF-8")
        other_data = open(fileother, 'w', encoding="UTF-8")

        # 写入文件
        for i, line in enumerate(lines):
            if i < other_num:
                other_data.write(line)
            elif i < n_total:
                fir_data.write(line)

        fir_data.close()
        other_data.close()


if __name__ == '__main__':
    merge_files("../cache/data_class", "../message/data/all_data.txt")
    # 将这些数据按照 8：2 的比例划分为训练集和测试集，再从划分出来的训练 集中按照 9:1 的比例划分为训练集和验证集
    # split("../data/all_data.txt", "../data/train_dev.txt", "../data/test.txt", 0.8, 0.2)
    # split("../data/train_dev.txt", "../data/train.txt", "../data/dev.txt", 0.9, 0.1)
    # 将数据按照测试集3000验证集1500划分
    split_num("../message/data/all_data.txt", "../message/data/train_dev.txt", "../message/data/test.txt", 3000)
    split_num("../message/data/train_dev.txt", "../message/data/train.txt", "../message/data/dev.txt", 1500)
