import torch
from ts.torch_handler.base_handler import BaseHandler
import json

class NILMHandler(BaseHandler):
    def preprocess(self, input):
        # 在此处对输入数据进行预处理
        # 这里假设输入数据是包含表格数据的列表
        print(input)
        mean = 5
        std = 3
        window_size = 480
        data = json.loads(input)['input']
        data = torch.tensor(data).reshape([-1,window_size])

        return (data - mean) / std

    def inference(self, data):
        # # 在此处进行推断
        # # 这里假设使用了一个预训练的模型进行推断
        # # 假设模型接受输入的形状为 (batch_size, num_features)
        # batch_size = len(data)
        # num_features = len(data[0])
        # input_tensor = torch.zeros((batch_size, num_features))  # 创建输入张量
        #
        # for i, item in enumerate(data):
        #     input_tensor[i] = torch.tensor(item)  # 将表格数据转换为张量

        # 使用预训练模型进行推断
        output = self.model(data)

        return output.tolist()

    def postprocess(self, data):
        # 在此处对推断结果进行后处理
        # 这里简单地返回推断结果
        print(data)
        return json.dumps({'result': data.detach().numpy().tolist()[0]})

# torchserve --start --model-store <model_store_path> --models <model_name>=<model_version>.mar
# torch-model-archiver --model-name BERT4NILM --version 1.0 --model-file BERT4NILM.py --serialized-file BERT4NILM.pt --handler NILMHandler.py --export-path ap/
