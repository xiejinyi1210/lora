import pandas as pd
import json  # 导入json模块，用于处理JSON数据

# 从Excel文件读取数据，转换为列表
data = pd.read_excel("input.xlsx").values.tolist()

result = []
for item in data:
    # 构造包含input和output的字典
    temp = {
        "input": item[0],
        "output": item[1]
    }
    result.append(temp)

# 将结果写入train.json文件，设置缩进、确保非ASCII字符正常显示
with open("train.json", "w", encoding="utf8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
