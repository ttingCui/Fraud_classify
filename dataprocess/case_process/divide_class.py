# # -*- coding = utf-8 -*-
# # 2023/5/5 17:15
#
# import json
#
# # 加载数据集
# with open('../../cache/case_data/train.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 获取类别标签
# def getclass(fileclass):
#     dic = {}
#     with open(fileclass, "r", encoding="UTF-8") as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#         # 获取类别 加载到词典中
#         for i, cls in enumerate(lines, 0):
#             if cls:
#                 dic[cls] = i
#     return dic

import json
import os

# 读取JSON文件
with open('../../cache/case_data/test_label.json', 'r', encoding="UTF-8") as f:
    data = json.load(f)

# 创建存储数据的目录
if not os.path.exists('../../cache/case_test_class'):
    os.makedirs('../../cache/case_test_class')

# 按案件类别将数据分组
grouped_data = {}
for item in data:
    category = item['案件类别']
    if category in grouped_data:
        # grouped_data[category].append(item['案情描述'])
        grouped_data[category].append(item['案件编号'])
    else:
        grouped_data[category] = [item['案件编号']]

# 将分组后的数据存储到文件中
for category, descriptions in grouped_data.items():
    with open(os.path.join('../../cache/case_test_class', f"{category}.txt"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(descriptions)))