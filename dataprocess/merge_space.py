# -*- coding = utf-8 -*-
# 2023/3/8 23:19
import os
# 合并分词
def delspace(fileindir, fileoutdir):
    for filename in os.listdir(fileindir):
        path = os.path.join(fileindir, filename)
        if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
            with open(path, "r", encoding="UTF-8") as f1, open(fileoutdir + "/" + filename, "w", encoding="UTF-8") as f2:
                for line in f1:
                    # line = line.replace("\n", "")
                    line = line.replace(" ", "")
                    f2.write(line)
                    # f2.write("\n")

delspace("../cache", "../cache/data_del_space")