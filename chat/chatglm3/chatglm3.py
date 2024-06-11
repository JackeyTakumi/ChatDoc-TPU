# coding=utf-8
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import configparser
from transformers import AutoTokenizer
import numpy as np
import time
import logging
import sophon.sail as sail

from ..utils import fp16_cast, type_convert
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class Chatglm3:
    def __init__(self, dev_id=0):
        config = configparser.ConfigParser()
        config.read('config.ini')
        bmodel_path = config.get('chatglm3', 'bmodel_path')
        token_path = config.get('chatglm3', 'token_path')
        # load tokenizer
        self.input_str = ""
        self.system = [{"role":"system",
                        "content":"You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."}]
        self.history = []
        self.sp = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
        logging.info("load {} success!".format(token_path))
        # warm up
        self.sp.decode([0])
        self.EOS = self.sp.eos_token_id

        # load bmodel
        # 这里devio，后面都没有创建系统内存的tensor
        self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success, dev_id {}".format(bmodel_path, dev_id))
        self.handle = sail.Handle(dev_id)
        self.graph_names = self.net.get_graph_names()

        # initialize glm parameters
        self.NUM_LAYERS = (len(self.graph_names) - 2) // 2
        self.first_hidden_input_shape = self.net.get_input_shape("block_0", self.net.get_input_names("block_0")[0])
        self.SEQLEN, _, self.HIDDEN_SIZE = self.first_hidden_input_shape

        self.name_embed = "embedding"
        self.name_embed_cache = "embedding_cache"
        self.name_lm = "lm_head"
        self.name_blocks = ["block_"+str(i) for i in range(self.NUM_LAYERS)]
        self.name_blocks_cache = ["block_cache_"+str(i) for i in range(self.NUM_LAYERS)]

        # tensors:
        # forward_first: embedding_tensor
        self.first_embed_input = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN])
        self.first_embed_output = self.init_sail_tensor(self.name_embed, 0, [1, self.SEQLEN, self.HIDDEN_SIZE], False)

        # forward_next: embedding_tensor
        self.next_embed_input = self.init_sail_tensor(self.name_embed_cache, 0, [1, 1])
        self.next_embed_output = self.init_sail_tensor(self.name_embed_cache, 0, [1,  1, self.HIDDEN_SIZE], False)

        # forward_first: hidden_state
        self.first_hidden_input = self.init_sail_tensor(self.name_blocks[0], 0)
        self.first_hidden_output = self.init_sail_tensor(self.name_blocks[0], 0, None, False)

        # forward_next: hidden_state
        self.next_hidden_input = self.init_sail_tensor(self.name_blocks_cache[0], 0)
        self.next_hidden_output = self.init_sail_tensor(self.name_blocks_cache[0], 0, None, False)

        # forward_first: position_id_tensor 和 attention_mask_tensor
        self.first_pid = self.init_sail_tensor(self.name_blocks[0], 1)
        self.first_attention = self.init_sail_tensor(self.name_blocks[0], 2)

        # forward_next: position_id_tensor and attention_mask_tensor
        self.next_pid = self.init_sail_tensor(self.name_blocks_cache[0], 1)
        self.next_attention = self.init_sail_tensor(self.name_blocks_cache[0], 2)

        # forward_next: present_key / present_value (for update kv_cache)
        self.present_key = self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False)
        self.present_value = self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False)

        # forward_first: key_tensor 和 value_tensor
        self.past_key_output = []
        self.past_value_output = []

        # forward_next: cache block的kv tensor名
        self.cache_key_input = []
        self.cache_key_output = []
        self.cache_value_input = []
        self.cache_value_output = []

        for i in range(self.NUM_LAYERS):
            self.past_key_output.append(self.init_sail_tensor(self.name_blocks[0], 1, None, False))
            self.past_value_output.append(self.init_sail_tensor(self.name_blocks[0], 2, None, False))
            self.past_key_output[i]["data"].memory_set(0)
            self.past_value_output[i]["data"].memory_set(0)

            self.cache_key_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 3))
            self.cache_key_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 1, None, False))

            self.cache_value_input.append(self.init_sail_tensor(self.name_blocks_cache[0], 4))
            self.cache_value_output.append(self.init_sail_tensor(self.name_blocks_cache[0], 2, None, False))

        # lm_head tensor
        self.lm_input = self.init_sail_tensor(self.name_lm, 0)
        self.lm_output = self.init_sail_tensor(self.name_lm, 0, None, False)

        self.token_length = 0
        self.round = 0

    def init_sail_tensor(self, name, tensor_idx, shape=None, is_input=True):
        """
        init a sail tensor of sail.engine.
        parameters:
        input:
            name: str, graph_name/net_name
            tensor_idx: int, input/output tensor id
            shape: list[int], shape of tensor
            is_input: bool, is input tensor or not
        return:
            dict
        """
        tensor = {}
        if is_input:
            tensor["name"] = self.net.get_input_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_input_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_input_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        else:
            tensor["name"] = self.net.get_output_names(name)[tensor_idx]
            tensor["shape"] = self.net.get_output_shape(name, tensor["name"]) if shape is None else shape
            tensor["dtype"] = self.net.get_output_dtype(name, tensor["name"])
            tensor["data"] = sail.Tensor(self.handle, tensor["shape"], tensor["dtype"], False, True)
        return tensor

    def generate_tokens(self, input_str):
        if not self.history or self.history[0]["role"] != "system":
            self.history = self.system + self.history
        tokens = self.sp.build_chat_input(input_str, history=self.history, role="user")
        return tokens

    def forward_first(self, token):
        # Keep
        input_ids = np.zeros(self.SEQLEN, type_convert(self.first_embed_input["dtype"]))
        input_ids[:min(self.SEQLEN, len(token))] = token
        self.token_length = len(token)
        input_ids = input_ids.reshape(1, -1)

        position_id = np.zeros(self.SEQLEN, type_convert(self.first_pid["dtype"]))
        for i in range(self.token_length):
            position_id[i] = i

        attention_mask = np.zeros(self.SEQLEN*self.SEQLEN, type_convert(self.first_attention["dtype"])) #这里的type要从模型获取。
        for i in range(self.SEQLEN):
            for j in range(self.SEQLEN):
                if not (j <= i and i < self.token_length):
                    attention_mask[i*self.SEQLEN + j] = -10000.0
        # embedding
        self.first_embed_input["data"].update_data(fp16_cast(input_ids))
        input_embed_tensors = {self.first_embed_input["name"]: self.first_embed_input["data"]}
        output_embed_tensors = {self.first_embed_output["name"]: self.first_embed_output["data"]}
        self.net.process(self.name_embed, input_embed_tensors, output_embed_tensors)

        # blocks
        self.first_hidden_tensor = self.first_embed_output["data"]
        self.first_hidden_tensor.reshape(self.first_hidden_input["shape"])
        self.first_pid["data"].update_data(fp16_cast(position_id.reshape(self.first_pid["shape"])))
        self.first_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.first_attention["shape"])))

        input_blocks_tensors = {self.first_hidden_input["name"]: self.first_hidden_tensor,
                                self.first_pid["name"]: self.first_pid["data"],
                                self.first_attention["name"]: self.first_attention["data"]}

        for i in range(self.NUM_LAYERS):
            output_blocks_tensors = {self.first_hidden_output["name"]: self.first_hidden_tensor,
                                    self.past_key_output[i]["name"]: self.present_key["data"],
                                    self.past_value_output[i]["name"]: self.present_value["data"],}
            self.net.process(self.name_blocks[i], input_blocks_tensors, output_blocks_tensors)

            unit_size = np.prod(self.present_key["shape"][1:])
            self.past_key_output[i]["data"].sync_d2d(self.present_key["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)
            self.past_value_output[i]["data"].sync_d2d(self.present_value["data"], 0, (self.SEQLEN - self.token_length)*unit_size, self.token_length * unit_size)

        # lm_head
        # hidden_states 的最后一个位置的元素取出来作为 lm_head的输入
        copy_len = self.first_hidden_tensor.shape()[-1]
        self.lm_input["data"].sync_d2d(self.first_hidden_tensor,
                                      (self.token_length-1)* copy_len,
                                      0,
                                      copy_len)

        input_lm_tensors = {self.lm_input["name"]: self.lm_input["data"]}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}

        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy())

    def forward_next(self, ):
        attention_mask = np.zeros(self.SEQLEN+1, type_convert(self.next_attention["dtype"]))
        for i in range(self.SEQLEN - self.token_length + 1):
            attention_mask[i] = -10000.0
        position_id = np.array(self.token_length - 1, type_convert(self.next_pid["dtype"]))

        # embedding
        self.next_embed_input["data"] = self.lm_output["data"]
        self.next_embed_input["data"].reshape(self.next_embed_input["shape"])

        input_embed_tensors = {self.next_embed_input["name"]: self.next_embed_input["data"]}
        output_embed_tensors = {self.next_embed_output["name"]: self.next_embed_output["data"]}
        self.net.process(self.name_embed_cache, input_embed_tensors, output_embed_tensors)

        # blocks
        self.next_pid["data"].update_data(fp16_cast(position_id.reshape(self.next_pid["shape"])))
        self.next_attention["data"].update_data(fp16_cast(attention_mask.reshape(self.next_attention["shape"])))

        self.next_hidden_tensor = self.next_embed_output["data"]
        self.next_hidden_tensor.reshape(self.next_hidden_input["shape"])

        for i in range(self.NUM_LAYERS):
            inputs_block_cache_tensors = {self.next_hidden_input["name"]: self.next_hidden_tensor,
                                        self.next_pid["name"]: self.next_pid["data"],
                                        self.next_attention["name"]: self.next_attention["data"],
                                        self.cache_key_input[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_input[i]["name"]: self.past_value_output[i]["data"]}
            outputs_block_cache_tensors = {self.next_hidden_output["name"]: self.next_hidden_tensor,
                                        self.cache_key_output[i]["name"]: self.past_key_output[i]["data"],
                                        self.cache_value_output[i]["name"]: self.past_value_output[i]["data"]}
            self.net.process(self.name_blocks_cache[i], inputs_block_cache_tensors, outputs_block_cache_tensors)

        self.lm_input_tensor = self.next_hidden_tensor
        self.lm_input_tensor.reshape(self.lm_input["shape"])

        input_lm_tensors = {self.lm_input["name"]: self.lm_input_tensor}
        output_lm_tensors = {self.lm_output["name"]: self.lm_output["data"]}
        self.net.process(self.name_lm, input_lm_tensors, output_lm_tensors)
        return int(self.lm_output["data"].asnumpy()) #int32


    def build_prompt(self, query, history):
        prompt = []
        # import pdb; pdb.set_trace()
        for i in range(0, len(history)):
            prompt.extend([{"role":"user", "content":history[i][0]},
                                 {"role":"assistant", "content":history[i][1]}])
        prompt = self.system + prompt
        prompt = self.sp.build_chat_input(query, history=prompt, role="user")
        return prompt

    def stream_predict(self, input_str, history):
        # import pdb; pdb.set_trace()
        prompt = self.build_prompt(input_str, history)
        history.append((input_str, ''))
        tok_num = 0
        answer_cur = []
        tokens = prompt
        # input is empty
        if not tokens:
            logging.error("Sorry: your question is too wierd!!")
            return
        if len(tokens) > self.SEQLEN:
            logging.warning("The maximum question length should be shorter than {} but we get {} instead, \
                history will be cleared, please ask again".format(self.SEQLEN, len(tokens)))
            return

        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()
        pre_token = 30910
        pre_ids = [pre_token]
        pre_word= self.sp.decode(pre_ids)
        # Sentencepiece will remove space token if the token list it receive has only one token, we add a pre_token so that space token will not be removed.
        while token != self.EOS and self.token_length < self.SEQLEN:
            ids = [pre_token, token]
            word = self.sp.decode(ids)
            diff = word[len(pre_word):]
            answer_cur += [token]
            yield self.sp.decode(answer_cur), history
            print(diff, flush=True, end='')
            self.token_length += 1
            tok_num += 1
            token = self.forward_next()

        # 计时
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
