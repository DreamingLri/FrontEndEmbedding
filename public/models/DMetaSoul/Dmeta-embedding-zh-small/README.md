<div align="center">
<img src="logo.png" alt="icon" width="100px"/>
</div>

<h1 align="center">Dmeta-embedding-small</h1>

- Dmeta-embedding系列模型是跨领域、跨任务、开箱即用的中文 Embedding 模型，适用于搜索、问答、智能客服、LLM+RAG 等各种业务场景，支持使用 Transformers/Sentence-Transformers/Langchain 等工具加载推理。
- **Dmeta-embedding-zh-small**是开源模型[Dmeta-embedding-zh](https://huggingface.co/DMetaSoul/Dmeta-embedding-zh)的蒸馏版本（8层BERT），模型大小不到300M。相较于原始版本，Dmeta-embedding-zh-small模型大小减小三分之一，推理速度提升约30%，总体精度下降约1.4%。

---

## Evaluation

这里主要跟蒸馏前对应的 teacher 模型作了对比：

*性能：*（基于1万条数据测试，GPU设备是V100）

|            | Teacher                   | Student                        | Gap   |
| ---------- | ------------------------- | ------------------------------ | ----- |
| Model      | Dmeta-Embedding-zh (411M) | Dmeta-Embedding-zh-small (297M)| 0.67x |
| Cost       | 127s                      | 89s                            | -30%  |
| Latency    | 13ms                      | 9ms                            | -31%  |
| Throughput | 78 sentence/s             | 111 sentence/s                 | 1.4x  |


*精度：*（参考自MTEB榜单）

|                               | **Classification** | **Clustering** | **Pair Classification** | **Reranking** | **Retrieval** | **STS** | **Avg** |
| ----------------------------- | -----------------  | -------------- | ----------------------- | ------------- | ------------- | ------- | ------- |
| **Dmeta-Embedding-zh**        | 70                 | 50.96          | 88.92                   | 67.17         | 70.41         | 64.89   | 67.51   |
| **Dmeta-Embedding-zh-small**  | 69.89              | 50.8           | 87.57                   | 66.92         | 67.7          | 62.13   | 66.1    | 
| **Gap**                       | -0.11              | -0.16          | -1.35                   | -0.25         | -2.71         | -2.76   | -1.41   |


## Usage

目前模型支持通过 [Sentence-Transformers](#sentence-transformers), [Langchain](#langchain), [Huggingface Transformers](#huggingface-transformers) 等主流框架进行推理，具体用法参考各个框架的示例。

### Sentence-Transformers

Dmeta-embedding 模型支持通过 [sentence-transformers](https://www.SBERT.net) 来加载推理：

```
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
texts1 = ["胡子长得太快怎么办？", "在香港哪里买手表好"]
texts2 = ["胡子长得快怎么办？", "怎样使胡子不浓密！", "香港买手表哪里好", "在杭州手机到哪里买"]
model = SentenceTransformer('DMetaSoul/Dmeta-embedding-zh-small')
embs1 = model.encode(texts1, normalize_embeddings=True)
embs2 = model.encode(texts2, normalize_embeddings=True)
# 计算两两相似度
similarity = embs1 @ embs2.T
print(similarity)
# 获取 texts1[i] 对应的最相似 texts2[j]
for i in range(len(texts1)):
    scores = []
    for j in range(len(texts2)):
        scores.append([texts2[j], similarity[i][j]])
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    print(f"查询文本：{texts1[i]}")
    for text2, score in scores:
        print(f"相似文本：{text2}，打分：{score}")
    print()
```

示例输出如下：

```
查询文本：胡子长得太快怎么办？
相似文本：胡子长得快怎么办？，打分：0.965681254863739
相似文本：怎样使胡子不浓密！，打分：0.7353651523590088
相似文本：香港买手表哪里好，打分：0.24928246438503265
相似文本：在杭州手机到哪里买，打分：0.2038613110780716

查询文本：在香港哪里买手表好
相似文本：香港买手表哪里好，打分：0.9916468262672424
相似文本：在杭州手机到哪里买，打分：0.498248815536499
相似文本：胡子长得快怎么办？，打分：0.2424771636724472
相似文本：怎样使胡子不浓密！，打分：0.21715955436229706
```

### Langchain

Dmeta-embedding 模型支持通过 LLM 工具框架 [langchain](https://www.langchain.com/) 来加载推理：

```
pip install -U langchain
```

```python
import torch
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
model_name = "DMetaSoul/Dmeta-embedding-zh-small"
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
texts1 = ["胡子长得太快怎么办？", "在香港哪里买手表好"]
texts2 = ["胡子长得快怎么办？", "怎样使胡子不浓密！", "香港买手表哪里好", "在杭州手机到哪里买"]
embs1 = model.embed_documents(texts1)
embs2 = model.embed_documents(texts2)
embs1, embs2 = np.array(embs1), np.array(embs2)
# 计算两两相似度
similarity = embs1 @ embs2.T
print(similarity)
# 获取 texts1[i] 对应的最相似 texts2[j]
for i in range(len(texts1)):
    scores = []
    for j in range(len(texts2)):
        scores.append([texts2[j], similarity[i][j]])
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    print(f"查询文本：{texts1[i]}")
    for text2, score in scores:
        print(f"相似文本：{text2}，打分：{score}")
    print()
```

### HuggingFace Transformers

Dmeta-embedding 模型支持通过 [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) 框架来加载推理：

```
pip install -U transformers
```

```python
import torch
from transformers import AutoTokenizer, AutoModel
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def cls_pooling(model_output):
    return model_output[0][:, 0]
texts1 = ["胡子长得太快怎么办？", "在香港哪里买手表好"]
texts2 = ["胡子长得快怎么办？", "怎样使胡子不浓密！", "香港买手表哪里好", "在杭州手机到哪里买"]
tokenizer = AutoTokenizer.from_pretrained('DMetaSoul/Dmeta-embedding-zh-small')
model = AutoModel.from_pretrained('DMetaSoul/Dmeta-embedding-zh-small')
model.eval()
with torch.no_grad():
    inputs1 = tokenizer(texts1, padding=True, truncation=True, return_tensors='pt')
    inputs2 = tokenizer(texts2, padding=True, truncation=True, return_tensors='pt')
    model_output1 = model(**inputs1)
    model_output2 = model(**inputs2)
    embs1, embs2 = cls_pooling(model_output1), cls_pooling(model_output2)
    embs1 = torch.nn.functional.normalize(embs1, p=2, dim=1).numpy()
    embs2 = torch.nn.functional.normalize(embs2, p=2, dim=1).numpy()
# 计算两两相似度
similarity = embs1 @ embs2.T
print(similarity)
# 获取 texts1[i] 对应的最相似 texts2[j]
for i in range(len(texts1)):
    scores = []
    for j in range(len(texts2)):
        scores.append([texts2[j], similarity[i][j]])
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    print(f"查询文本：{texts1[i]}")
    for text2, score in scores:
        print(f"相似文本：{text2}，打分：{score}")
    print()
```
## Contact
您如果在使用过程中，遇到任何问题，欢迎前往[讨论区](https://huggingface.co/DMetaSoul/Dmeta-embedding-zh-small/discussions)建言献策。
您也可以联系我们：赵中昊 <zhongh@dmetasoul.com>, 肖文斌 <xiaowenbin@dmetasoul.com>, 孙凯 <sunkai@dmetasoul.com>
同时我们也开通了微信群，可扫码加入我们（人数超200了，先加管理员再拉进群），一起共建 AIGC 技术生态！
<image src="https://huggingface.co/DMetaSoul/Dmeta-embedding-zh-small/resolve/main/weixin.jpeg" style="display: block; margin-left: auto; margin-right: auto; width: 256px; height: 358px;"/>
## License
Dmeta-embedding 系列模型采用 Apache-2.0 License，开源模型可以进行免费商用私有部署。