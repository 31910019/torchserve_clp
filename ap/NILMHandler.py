import torch
from ts.torch_handler.base_handler import BaseHandler
import json
import logging
class NILMHandler(BaseHandler):
    def __init__(self):
        super(NILMHandler, self).__init__()
    
    def preprocess(self, input):
        # 在此处对输入数据进行预处理
        # 这里假设输入数据是包含表格数据的列表
        payload = {}
        logging.info('Receiving new data')

        payload["invalid_request"] = False
        try:
            req = input[0].get("body")
            req = json.loads(req)
            data = req.get("input", None)
            # logging.info("Received x: {}".format(data))
        except Exception as e:
            logging.info(f"Bad request...{e}")

        # logging.info(
            # f'''Going to inference - data: {data}'])''')
        mean = 5
        std = 3
        self.window_size = 144
        data = torch.tensor(data).reshape([-1,self.window_size])

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
        output = self.model(data.to(self.device))

        return output.reshape(-1,self.window_size)

    def postprocess(self, data):
        # 在此处对推断结果进行后处理
        # 这里简单地返回推断结果
        print(type(data))
        return data.cpu().detach().numpy().tolist()

# torchserve --start --model-store <model_store_path> --models <model_name>=<model_version>.mar
# torch-model-archiver --model-name BERT4NILM --version 1.0 --model-file BERT4NILM.py --serialized-file BERT4NILM.pt --handler NILMHandler.py --export-path ap/
_service = NILMHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    logging.info(f"Returning data: {data}")

    return data
