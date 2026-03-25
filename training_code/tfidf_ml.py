# model/tfidf_ml.py
import os
import jieba
import pandas as pd
from typing import Union, List
from joblib import load

# 导入配置
from config import (
    STOPWORDS_PATH,
    TFIDF_MODEL_PATH,
    CATEGORY_NAMES
)
from logger import logger

# 1. 启动时预加载模型和停用词（只加载一次）
logger.info("正在预加载 TF-IDF 模型...")

# 加载停用词
cn_stopwords = set(pd.read_csv(STOPWORDS_PATH, header=None)[0].values)

# 加载模型
if not os.path.exists(TFIDF_MODEL_PATH):
    raise FileNotFoundError(f"找不到模型文件：{TFIDF_MODEL_PATH}，请先运行 train_tfidf.py！")

tfidf_vectorizer, model = load(TFIDF_MODEL_PATH)
logger.info("✅ TF-IDF 模型预加载完成！")


def model_for_tfidf(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    对外暴露的推理函数
    :param request_text: 可以是一个字符串，也可以是一个字符串列表
    :return: 对应的分类结果
    """

    # 统一预处理逻辑
    def _preprocess(text):
        words = jieba.lcut(text)
        words = [w for w in words if w not in cn_stopwords]
        return " ".join(words)

    # 情况1：输入是单个字符串
    if isinstance(request_text, str):
        text_clean = _preprocess(request_text)
        # 注意：sklearn 要求输入是列表，所以要包一层 [text_clean]
        text_vec = tfidf_vectorizer.transform([text_clean])
        # 模型返回的是数组，取第0个
        result = model.predict(text_vec)[0]
        return result

    # 情况2：输入是列表（批量预测）
    elif isinstance(request_text, list):
        texts_clean = [_preprocess(t) for t in request_text]
        texts_vec = tfidf_vectorizer.transform(texts_clean)
        results = list(model.predict(texts_vec))
        return results

    else:
        raise TypeError("输入格式不支持，请输入 str 或 List[str]")


# --------------------------
# 测试代码
# --------------------------
if __name__ == "__main__":
    test_sentences = [
        "播放一首周杰伦的歌",
        "明天深圳天气怎么样",
        "你好"
    ]

    print("\n" + "=" * 60)
    print("单条测试:")
    print(f"输入: {test_sentences[0]}")
    print(f"输出: {model_for_tfidf(test_sentences[0])}")

    print("\n批量测试:")
    results = model_for_tfidf(test_sentences)
    for sent, res in zip(test_sentences, results):
        print(f"输入: {sent:20s} -> 输出: {res}")
    print("=" * 60 + "\n")