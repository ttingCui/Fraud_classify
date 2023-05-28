# import json
#
# # 读取JSON文件
# with open('../../cache/case_data/train.json', 'r', encoding="UTF-8") as f:
#     data = json.load(f)
#
# # 统计每个类别的数据量
# category_count = {}
# for item in data:
#     category = item['案件类别']
#     if category in category_count:
#         category_count[category] += 1
#     else:
#         category_count[category] = 1
#
# # 打印每个类别的数量
# for category, count in category_count.items():
#     print(f"{category}: {count}")


# 刷单返利类: 28367
# 虚假网络投资理财类: 9469
# 冒充电商物流客服类: 11018
# 贷款、代办信用卡类: 8883
# 网络游戏产品虚假交易类: 1723
# 虚假购物、服务类: 5647
# 冒充公检法及政府机关类: 3651
# 网黑案件: 958
# 虚假征信类: 6771
# 冒充领导、熟人类: 3525
# 冒充军警购物类诈骗: 874
# 网络婚恋、交友类（非虚假网络投资理财类）: 1324

