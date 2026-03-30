# model/prompt.py
import os
import numpy as np
import pandas as pd
from typing import Union, List
from joblib import load
import openai

# 导入配置和日志
from config import DATA_PATH, TFIDF_MODEL_PATH, CATEGORY_NAMES
from logger import logger

# ==========================================
# 📍 LLM 基础配置 (原项目使用的是本地 Ollama 部署的模型)
# ==========================================
# 如果你本地没有安装 Ollama，后面我们可以换成真实的 API Key (如 Kimi, 智谱等)
LLM_OPENAI_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_OPENAI_API_KEY = "sk-40ddce944f9c41f9bd7329bbbde37c3f"  # 本地大模型不需要真实key
LLM_MODEL_NAME = "qwen-plus"  # 原项目使用的是千问 0.5B 小模型

# ==========================================
# 🛠️ 第一部分：初始化“检索库”(Knowledge Base)
# ==========================================
logger.info("正在初始化大模型 RAG 检索知识库...")

# 1. 加载所有的历史数据（作为大模型的参考题库）
if not os.path.exists(DATA_PATH) or not os.path.exists(TFIDF_MODEL_PATH):
    logger.warning("找不到数据集或 TF-IDF 模型，LLM 检索可能失败！")
else:
    # 读出题库中的问题和答案
    train_data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text', 'label'])

    # 2. 把题库全部向量化（变成数学矩阵），用于计算相似度
    tfidf_vectorizer, _ = load(TFIDF_MODEL_PATH)
    # 获取整个题库的 TF-IDF 矩阵
    train_tfidf_matrix = tfidf_vectorizer.transform(train_data['text'])

# 3. 初始化 OpenAI 格式的客户端
client = openai.Client(base_url=LLM_OPENAI_SERVER_URL, api_key=LLM_OPENAI_API_KEY)

# 大模型的系统提示词模板 (Prompt Template)
PROMPT_TEMPLATE = '''你是一个意图识别的专家，请严格结合待选类别和参考例子进行意图分类。
待选类别：{categories}

历史参考例子如下：
{examples}

待识别的文本为：{query}
只需要输出意图类别（必须从待选类别中选一个），绝不要输出任何解释或其他内容。'''


# ==========================================
# 🚀 第二部分：大模型预测核心逻辑
# ==========================================
def model_for_gpt(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    大模型 Few-shot 推理接口
    """
    is_single = isinstance(request_text, str)
    texts = [request_text] if is_single else request_text
    classify_results = []

    # 1. 将用户当前的提问转为向量
    query_tfidf_matrix = tfidf_vectorizer.transform(texts)

    # 2. 遍历每个用户的提问
    for idx, query_text in enumerate(texts):
        # ---------- RAG 核心步骤 1：检索 (Retrieval) ----------
        # 计算当前句子和题库里所有句子的余弦相似度 (矩阵点乘)
        similarity_scores = np.dot(query_tfidf_matrix[idx], train_tfidf_matrix.T).toarray()[0]

        # 找出分数最高的 Top 5 句子的索引
        top5_indices = similarity_scores.argsort()[::-1][:5]

        # ---------- RAG 核心步骤 2：组装提示词 (Augmented) ----------
        dynamic_examples = ""
        for i in top5_indices:
            sim_text = train_data.iloc[i]['text']
            sim_label = train_data.iloc[i]['label']
            dynamic_examples += f"文本: {sim_text} -> 意图: {sim_label}\n"

        # 拼接最终发送给大模型的话
        categories_str = "/".join(CATEGORY_NAMES)
        final_prompt = PROMPT_TEMPLATE.format(
            categories=categories_str,
            examples=dynamic_examples,
            query=query_text
        )

        logger.info(f"生成的 Prompt 预览:\n{final_prompt}")

        # ---------- RAG 核心步骤 3：生成 (Generation) ----------
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "user", "content": final_prompt},
                ],
                temperature=0.0,  # 设为 0，保证模型不瞎编，只做稳定分类
                max_tokens=64,
            )
            result = response.choices[0].message.content.strip()
            classify_results.append(result)
        except Exception as e:
            logger.error(f"大模型调用失败: {e}")
            classify_results.append("Other")

    return classify_results[0] if is_single else classify_results


# ==========================================
# 🧪 第三部分：本地测试
# ==========================================
if __name__ == "__main__":
    # 注意：运行这个测试前，你需要有一个兼容 OpenAI 格式的大模型服务在运行。
    # 比如本地启动了 Ollama，或者修改配置换成阿里云/智谱的 API。
    test_query = "你能帮我把卧室的空调打开吗"

    print("\n" + "=" * 60)
    print("开始 LLM 大模型推理测试：")
    res = model_for_gpt(test_query)
    print(f"\n最终预测意图: {res}")
    print("=" * 60 + "\n")