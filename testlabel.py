from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# labels_list = ["贷款",
#                "流量",
#                "报名",
#                "房产",
#                "购货",
#                "基金",
#                "航班",
#                "信金",
#                "详查",
#                "服务",
#                "证票",
#                "彩金",
#                "中党",
#                "个人"]
labels_list = [x.strip() for x in open('message/data/class_tfidf.txt', encoding="UTF-8").readlines()]
# labels_list = ["贷款",
#                "网络",
#                "广告",
#                "房产",
#                "零售",
#                "金融",
#                "欺诈",
#                "银行",
#                "钓鱼",
#                "陪护",
#                "证票",
#                "赌博",
#                "政治",
#                "私人"]
# labels_list = ["贷款广告", "网络广告", "其他广告", "房产广告", "零售广告", "金融欺诈", "其他欺诈", "银行钓鱼", "非法钓鱼", "非法陪护", "假证假票", "非法赌博", "非法政治", "私人交流"]

# with open(filename, 'r', encoding='UTF-8') as f:
with open("message/label_id/label_id_tfidf.txt", 'w', encoding="UTF-8") as f:
    for label in labels_list:
        label_id = tokenizer.encode_plus(label, add_special_tokens=False)
        # print(label_id)
        for id in label_id["input_ids"]:
            f.write(str(id))
            f.write(" ")
        f.write("\n")