from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import torch
import numpy as np
import os

model_name = '../bge-reranker-large'

model = CrossEncoder(model_name=model_name, max_length=1024, device='cpu')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_name, revision = None, local_files_only = False, trust_remote_code = False)

texts = [['如何解决rtsp问题', '如何解决rtsp问题', '如何解决rtsp问题'], ['RTSP拉流解码卡住，不报错 多见于大华、海康的RTSP 如果RTSP的地址有些特殊符号，形如 rtsp://[username]:[password]@[ip]:[port]/cam/playback?channel=1&subtype=0&starttime=2017_01_10_01_00_00&endtime=2017_01_10_02_00_00 ， 要注意命令行解析字符串的时候可能出错 解决方法是给RTSP地址加上引号 "rtsp://..." 解码失败的保底排查方案', '解决方法是给RTSP地址加上引号 "rtsp://..." 解码失败的保底排查方案 保存原始码流，拿到本地分析码流。有以下两种保存方式。 ffmpeg -rtsp_transport tcp -i rtsp://xxx -c copy -vframes 500 saved.264 sudo echo "0 0 1000 0" > /proc/vpuinfo 参数1：0 core idx；参数2：0 instance idx；参数3：输入num（保存的帧数）；参数4：输出num（保存的yuv）', '-rtsp_transport tcp。 网络带宽较小，多路解码带宽占满，造成丢包。 设备解码能力到达上限了。这里说的“上限”，未必是VPU本身的硬件性能到上限了。比如，有可能是上层代码没有合理利用CPU等资源，导致VPU不能完全发挥性能。比如，如果使用ffmpeg命令行执行RTSP拉流解码+编码，而没有设置硬编码，因为A53编码较慢，编码阻塞，进而影响解码的调度，出现数据丢失。这时修改命令调用VPU硬编码可以解决。 推流不稳定。可以用VLC播放，观察是否有花屏。 Invalid NAL']]
tokenized = tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=1024
        )
input_ids = tokenized['input_ids']
attention_mask = tokenized['attention_mask']

input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()
if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=1)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
    
folder_name = "onnx"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
    
np.savez("./onnx/test_input.npz", input_ids=input_ids, attention_mask=attention_mask)
input_ids = torch.tensor(input_ids)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    model_predictions = model.model(input_ids, attention_mask, return_dict=True)


torch.onnx.export(model.model, (input_ids,attention_mask), "./onnx/bge_reranker_large.onnx", input_names=['input_ids', 'attention_mask'], dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}})


print("load model done")