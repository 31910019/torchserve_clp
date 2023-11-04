import json
import numpy as np
list_data = np.random.randn(1,480).tolist()
json_data = json.dumps(list_data)

import requests
import json

# 请求数据和 API 路径
data = {"input": "your_input_data"}
url = "http://127.0.0.1:8080"

# 发送 POST 请求到 TorchServe 的 API 路径
response = requests.post(url, json=json_data)

# 处理响应
if response.status_code == 200:
    result = response.json()
    # 处理结果
    print(result)
else:
    print("API request failed with status code:", response.status_code)