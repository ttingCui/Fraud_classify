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
    # get_fewshot_data("../../cache/data_class", "../../message/fewshotdata_8/train_tmp.txt", "../../message/fewshotdata_8/dev_tmp.txt", "../../message/fewshotdata_8/test_tmp.txt", class_num=8)
    # random_file("../../message/fewshotdata_8/train_tmp.txt", "../../message/fewshotdata_8/train.txt")
    # random_file("../../message/fewshotdata_8/dev_tmp.txt", "../../message/fewshotdata_8/dev.txt")
    # random_file("../../message/fewshotdata_8/test_tmp.txt", "../../message/fewshotdata_8/test.txt")

    # get_fewshot_data("../../cache/data_class", "../../message/fewshotdata_16/train_tmp.txt", "../../message/fewshotdata_16/dev_tmp.txt", "../../message/fewshotdata_16/test_tmp.txt")
    # random_file("../../message/fewshotdata_16/train_tmp.txt", "../../message/fewshotdata_16/train.txt")
    # random_file("../../message/fewshotdata_16/dev_tmp.txt", "../../message/fewshotdata_16/dev.txt")
    # random_file("../../message/fewshotdata_16/test_tmp.txt", "../../message/fewshotdata_16/test.txt")

    # get_fewshot_data("../../cache/data_class", "../../message/fewshotdata_32/train_tmp.txt",
    #                  "../../message/fewshotdata_32/dev_tmp.txt", "../../message/fewshotdata_32/test_tmp.txt", class_num=32)
    # random_file("../../message/fewshotdata_32/train_tmp.txt", "../../message/fewshotdata_32/train.txt")
    # random_file("../../message/fewshotdata_32/dev_tmp.txt", "../../message/fewshotdata_32/dev.txt")
    # random_file("../../message/fewshotdata_32/test_tmp.txt", "../../message/fewshotdata_32/test.txt")

    get_fewshot_data("../../cache/data_class", "../../message/fewshotdata_64/train_tmp.txt",
                     "../../message/fewshotdata_64/dev_tmp.txt", "../../message/fewshotdata_64/test_tmp.txt", class_num=64)
    random_file("../../message/fewshotdata_64/train_tmp.txt", "../../message/fewshotdata_64/train.txt")
    random_file("../../message/fewshotdata_64/dev_tmp.txt", "../../message/fewshotdata_64/dev.txt")
    random_file("../../message/fewshotdata_64/test_tmp.txt", "../../message/fewshotdata_64/test.txt")
