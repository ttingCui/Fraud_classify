# -*- coding = utf-8 -*-
# 2023/3/14 16:17

# -*- coding = utf-8 -*-
# 2023/3/8 22:58
import os, random

# 每个类别获取少量的样本数据
def get_fewshot_data(filedir, filetrain, filedev, filetest, class_num=16):
    with open(filetrain, "w", encoding="UTF-8") as f1, open(filedev, "w", encoding="UTF-8") as f2, open(filetest, "w", encoding="UTF-8") as f3:
        for filename in os.listdir(filedir):
            path = os.path.join(filedir, filename)
            if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
                with open(path, "r", encoding="UTF-8") as f4:
                    lines = f4.readlines()
                    random.shuffle(lines)  # 按行打乱顺序
                    for i, line in enumerate(lines):
                        if i < class_num:
                            f1.write(line)
                        elif i < class_num*2:
                            f2.write(line)
                        else:
                            f3.write(line)


def random_file(filein, fileout):
    with open(filein, "r", encoding="UTF-8") as f1, open(fileout, "w", encoding="UTF-8") as f2:
        lines = f1.readlines()
        random.shuffle(lines)
        f2.write("".join(lines))



if __name__ == '__main__':
    # get_fewshot_data("../../cache/data_class", "../../message/fewshotdata/train_tmp.txt", "../../message/fewshotdata/dev_tmp.txt", "../../message/fewshotdata/test_tmp.txt")
    random_file("../../message/fewshotdata/train_tmp.txt", "../../message/fewshotdata/train.txt")
    random_file("../../message/fewshotdata/dev_tmp.txt", "../../message/fewshotdata/dev.txt")
    random_file("../../message/fewshotdata/test_tmp.txt", "../../message/fewshotdata/test.txt")
    # 将数据按照测试集3000验证集1500划分
    # split_num("../message/data/all_data.txt", "../message/data/train_dev.txt", "../message/data/test.txt", 3000)
    # split_num("../message/data/train_dev.txt", "../message/data/train.txt", "../message/data/dev.txt", 1500)
