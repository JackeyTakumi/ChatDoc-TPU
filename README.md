# ChatDoc-TPU <!-- omit in toc -->

这个项目是基于 Sophgo TPU 实现的文档对话工具。项目可在 BM1684X 上独立部署运行。

- [介绍](#介绍)
- [特点](#特点)
- [安装](#安装)
  - [安装第三方库](#安装第三方库)
  - [安装sail](#安装sail)
- [项目结构树](#项目结构树)
- [启动](#启动)
- [操作说明](#操作说明)
  - [界面简介](#界面简介)
  - [上传文档](#上传文档)
  - [持久化知识库](#持久化知识库)
  - [导入知识库](#导入知识库)
  - [删除知识库](#删除知识库)
  - [重命名知识库](#重命名知识库)
  - [清除聊天记录](#清除聊天记录)
  - [移除选中文档](#移除选中文档)


## 介绍

该项目的主要目标是通过使用自然语言来简化与文档的交互，并提取有价值的信息。此项目使用LangChain、[ChatGLM3-TPU](https://github.com/sophgo/sophon-demo/tree/release/sample/ChatGLM3)或[QWEN-TPU](https://github.com/sophgo/sophon-demo/tree/release/sample/Qwen)构建，以向用户提供流畅自然的对话体验。

以 ChatGPT 为例（可替换为其他LLM，本仓库已支持 Chatglm3-6B 和 Qwen-7B，需要保证接口一致），本地知识库问答流程如下：
![Flow](<./static/embedding.png>)

## 特点

- 完全本地推理。
- 支持多种文档格式PDF, DOCX, TXT。
- 与文档内容进行聊天，提出问题根据文档获得相关答案。
- 用户友好的界面，确保流畅的交互。


## 安装

按照以下步骤，可以将这个项目部署到SophGo的设备上

### 安装第三方库
```bash
cd ChatDoc-TPU
# 考虑到 langchain 和 sail 版本依赖，推荐在 python>=3.8 环境运行
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 安装sail

此例程`依赖新版本sail`，旧版本需要更新，安装方法请参考[Sail_Install_Guide](./docs/Sail_Install_Guide.md)

## 项目结构树
```
|-- ChatDoc-TPU
    |-- data
        |-- db_tpu        -- 知识库持久化目录
        |-- uploaded      -- 已上传文件目录
    |-- models
        |-- bert_model    -- BERT 模型
        |-- glm3_model    -- charglm3-6B 模型
        |-- qwen_model    -- qwen-7B 模型
    |-- chat
        |-- chatbot.py    -- ChatDoc业务逻辑脚本
        |-- charglm3      -- charglm3 代码
        |-- qwen          -- qwen 代码
    |-- embedding         -- 文本嵌入模型
    |-- docs              -- 环境安装文档
    |-- static            -- README中图片文件
    |-- api.py            -- API服务脚本
    |-- README.md         -- README
    |-- config.ini        -- 推理模型配置文件
    |-- requirements.txt  -- 项目依赖
    |-- run.sh            -- 启动脚本
    |-- web_demo_st.py    -- 页面交互脚本
```

## 启动

回到`ChatDoc-TPU`主目录，启动程序，模型和配置文件自动下载，使用默认路径

| Model           | Cmd                                  |
| :-------------- | :------------------------------------|
| ChatGLM3-6B     | ./run.sh --model chatglm3 --dev_id 0 |
| Qwen-7B         | ./run.sh --model qwen --dev_id 0     |

```bash
usage: ./run.sh [--model MODEL]  [--dev_id DEV_ID] [--server_address SERVER_ADDRESS] [--server_port SERVER_PORT]
--model: 选择模型，可选项为 chatglm3/qwen。默认为 "chatglm3"。
--dev_id: 用于推理的 TPU 设备 ID。默认为 0。
--server_address: web server 地址。默认为 "0.0.0.0"。
--server_port：web sever 端口。如不设置，从 8501 起自动分配。
```

启动后您可以通过浏览器打开，`URL: http://{host_ip}:8501`，host_ip为启动ChatDoc的设备IP，或者您通过参数设置的`server_address`

> **说明**：
>1. 在 `config.ini` 中可修改模型路径，默认使用int4模型
>2. dev_id 需设置为 BM1684X 设备id
>3. 默认使用 2k seq_len 模型，如果需要其他参数的模型，可参考[ChatGLM3模型导出与编译](https://github.com/sophgo/sophon-demo/blob/release/sample/ChatGLM3/docs/ChatGLM3_Export_Guide.md)和[Qwen模型导出与编译](https://github.com/sophgo/sophon-demo/blob/release/sample/Qwen/docs/Qwen_Export_Guide.md)
>4. embedding 模型默认使用 [shibing624/text2vec-bge-large-chinese](https://huggingface.co/shibing624/text2vec-bge-large-chinese)，导出模型方法可参考 [export_onnx.py](./scripts/export_onnx.py)

如果您只想启动以后后端api接口，可以在下载完毕模型之后，运行
```bash
python api.py
```

## 操作说明

![UI](<./static/img1.png>)

### 界面简介
ChatDoc由控制区和聊天对话区组成。控制区用于管理文档和知识库，聊天对话区用于输入和接受消息。

上图中的10号区域是 ChatDoc 当前选中的文档。若10号区域为空，即 ChatDoc 没有选中任何文档，仍在聊天对话区与 ChatDoc 对话，则此时的 ChatDoc 是一个单纯依托 LLM 的 ChatBot。

### 上传文档
点击`1`选择要上传的文档，然后点击按钮`4`构建知识库。随后将embedding文档，完成后将被选中，并显示在10号区域，接着就可开始对话。我们可重复上传文档，embedding成功的文档均会进入10号区域。

### 持久化知识库
10号区域选中的文档在用户刷新或者关闭页面时，将会清空。若用户需要保存这些已经embedding的文档，可以选择持久化知识库，下次进入时无需embedding计算即可加载知识库。具体做法是，在10号区域不为空的情况下，点击按钮`5`即可持久化知识库，知识库的名称是所有文档名称以逗号连接而成。

### 导入知识库

用户可以从选择框`2`查看目前已持久化的知识库。选中我们需要加载的知识库后，点击按钮`3`导入知识库。完成后即可开始对话。注意cpu版的知识库和tpu版的知识库不能混用，若启动tpu版程序，则不能加载已持久化的cpu版知识库；若启动cpu版程序，则不能加载已持久化的tpu版知识库。

### 删除知识库

当用户需要删除本地已经持久化的知识库时，可从选择框`2`选择要删除的知识库，然后点击按钮`6`删除知识库。

### 重命名知识库

![Rename](<./static/img2.png>)

由于知识库的命名是由其文档的名称组合而来，难免造成知识库名称过长的问题。ChatDoc提供了一个修改知识库名称的功能，选择框`2`选择我们要修改的知识库，然后点击按钮`9`重命名知识库，随后ChatDoc将弹出一个输入框和一个确认按钮，如上图。在输出框输入修改后的名称，然后点击`确认重命名`按钮。

### 清除聊天记录

点击按钮`7`即可清除聊天对话区聊天记录。其他不受影响。

### 移除选中文档

点击按钮`8`将清空10号区域，同时清除聊天记录。