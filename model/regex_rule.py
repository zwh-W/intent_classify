# model/regex_rule.py
import re
from typing import Union, List
from config import CATEGORY_NAMES
from logger import logger

# 定义简单的关键词规则（只写几个作为示例）
KEYWORD_RULES = {
    "Music-Play": ["播放", "歌", "音乐"],
    "Weather-Query": ["天气", "温度", "下雨"],
    "Alarm-Update": ["闹钟", "提醒", "定"],
    "FilmTele-Play": ["电视剧", "电影", "视频"]
}

# 预编译正则，提高速度
COMPILED_RULES = {}
for category, keywords in KEYWORD_RULES.items():
    COMPILED_RULES[category] = re.compile("|".join(keywords))


def model_for_regex(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    基于关键词规则的分类
    """

    # 内部分类函数
    def _classify_single(text):
        for category, pattern in COMPILED_RULES.items():
            if pattern.search(text):
                return category
        # 如果没匹配到任何关键词，返回 Other
        return "Other"

    if isinstance(request_text, str):
        return _classify_single(request_text)

    elif isinstance(request_text, list):
        return [_classify_single(t) for t in request_text]

    else:
        raise TypeError("输入格式不支持")


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
    print("规则模型测试:")
    results = model_for_regex(test_sentences)
    for sent, res in zip(test_sentences, results):
        print(f"输入: {sent:20s} -> 输出: {res}")
    print("=" * 60 + "\n")