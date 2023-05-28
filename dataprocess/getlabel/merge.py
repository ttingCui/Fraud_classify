# -*- coding = utf-8 -*-
# 2023/5/28 20:15

import os
import re


# 处理每个文件的数据为一行，合并所有文件
def merge_data_from_folders(filedir, output_file):
    with open(output_file, "w", encoding="UTF-8") as outfile:
        for filename in os.listdir(filedir):
            path = os.path.join(filedir, filename)
            if os.path.isfile(path):  # 判断是否是文件还是目录需要用绝对路径
                with open(path, "r", encoding="UTF-8") as infile:
                    lines = infile.readlines()
                    # lines_ =
                    merged_data = " ".join(" ".join(line.strip().split()) for line in lines)
                    # merged_data = " ".join(line.strip() for line in lines)
                    # data = infile.read().strip()
                    outfile.write(f"{filename} {merged_data} ")
                outfile.write("\n")


# 每个中文字符之间增加空格
def add_data_space(fileindir, fileoutdir):
    for filename in os.listdir(fileindir):
        path = os.path.join(fileindir, filename)
        with open(path, "r", encoding="UTF-8") as f1, open(fileoutdir+"/"+filename, "w", encoding="UTF-8") as f2:
            for line in f1:
                # lines = infile.readlines()
                characters = [char for char in line]
                line = " ".join(characters)
                f2.write(line)
                # f2.write("\n")


def extract_top_keywords(data_file):
    top_keywords = {}

    with open(data_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    current_category = None
    for line in lines:
        line = line.strip()
        if line:
            if line.endswith(":"):
                current_category = line[:-1]
                top_keywords[current_category] = []
            else:
                keyword, score = line.split(":")
                keyword = keyword.strip()
                score = float(score.strip())
                # if not re.match(r'^[a-zA-Z]+$', keyword):  # 判断关键字不只包含字母
                if re.search(r'[\u4e00-\u9fff]', keyword):  # top_keywords.append((keyword, score))
                # if keyword.isalpha():
                #     continue  # 舍弃字母
                    top_keywords[current_category].append((keyword, score))

    # 对每个类别的关键字按得分进行排序，并选取前10个
    for category in top_keywords:
        top_keywords[category] = sorted(top_keywords[category], key=lambda x: x[1], reverse=True)[:10]

    return top_keywords

# 示例用法
# if __name__ == "__main__":
#     data_file = "data.txt"  # 替换为你的数据文件路径
#     top_keywords = extract_top_keywords(data_file)
#     for category, keywords in top_keywords.items():
#         print(f"{category}:")
#         for keyword, score in keywords:
#             print(f"{keyword}: {score}")
#         print()



# 示例用法
if __name__ == "__main__":
    # add_data_space("../../cache/data_del_space", "../../cache/message_space")
    # folders = "../../cache/message_space"  # 替换为你的文件夹路径列表
    # output_file = "../../cache/merged_data.txt"  # 合并后的输出文件路径
    # merge_data_from_folders(folders, output_file)
    data_file = "../../cache/get_label/tfidf_model.txt"  # 替换为你的数据文件路径
    output_file = "../../cache/get_label/tfidf.txt"  # 替换为你的数据文件路径
    top_keywords = extract_top_keywords(data_file)
    with open(output_file, "w", encoding="utf-8") as file:
        for category, keywords in top_keywords.items():
            # file.write(category + ":\n")
            for keyword, score in keywords:
                file.write(keyword)
                # file.write(keyword + "\n")
            file.write("\n")
    # for category, keywords in top_keywords.items():
    #     print(f"{category}:")
    #     for keyword, score in keywords:
    #         print(f"{keyword}: {score}")
    #     print()
