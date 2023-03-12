# -*- coding = utf-8 -*-
# 2023/3/8 22:58
import os




# 在数据集中加入类别标签
def addclass(classdic, fileindir, fileoutdir):
    for filename in os.listdir(fileindir):
        path = os.path.join(fileindir, filename)
        with open(path, "r", encoding="UTF-8") as f1, open(fileoutdir+"/"+filename, "w", encoding="UTF-8") as f2:
            for line in f1:
                line = line.replace("\n", "")
                f2.write(line)
                f2.write("\t")
                f2.write(str(classdic[filename]))
                f2.write("\n")


# 获取类别标签
def getclass(fileclass):
    dic = {}
    with open(fileclass, "r", encoding="UTF-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        # 获取类别 加载到词典中
        for i, cls in enumerate(lines, 0):
            if cls:
                dic[cls] = i
    return dic



if __name__ == '__main__':
    class_dic = getclass("../message/data/class.txt")
    addclass(class_dic, "../cache/data_del_space", "../cache/data_class")