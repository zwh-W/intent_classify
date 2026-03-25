# model/bert.py
import os
import torch
import numpy as np
from typing import Union, List
from transformers import BertTokenizer, BertForSequenceClassification

# 导入我们的基础配置
from config import BASE_DIR, CATEGORY_NAMES
from logger import logger

# ==========================================
# 📍 核心路径配置
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'models', 'AI-ModelScope', 'bert-base-chinese')
BERT_MODEL_PT = os.path.join(BASE_DIR, "assets", "weights", "bert.pt")

# ==========================================
# 🛠️ 第一部分：系统启动时的初始化 (只执行一次)
# ==========================================
# 1. 确定用 CPU 还是 GPU (如果有独显会自动用显卡加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"正在加载 BERT 推理引擎，使用设备: {device} ...")

try:
    # 2. 加载字典 (Tokenizer)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 3. 加载大脑结构 (声明这是个 12 分类的任务)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(CATEGORY_NAMES))

    # 4. 核心：注入灵魂！把你刚刚辛辛苦苦训练出来的 bert.pt 权重加载进大脑
    # map_location=device 确保即便你在GPU上训练，换到CPU机器上也能正常加载
    model.load_state_dict(torch.load(BERT_MODEL_PT, map_location=device, weights_only=True))

    # 5. 把模型放到对应设备，并开启"考试模式" (eval)
    model.to(device)
    model.eval()  # 关掉 Dropout 等训练机制，保证预测稳定
    logger.info("✅ BERT 推理引擎加载成功！时刻准备接客！")
except Exception as e:
    logger.warning(f"⚠️ BERT 加载未完成: 可能是 bert.pt 还没训练完，先忽略。报错信息: {e}")


# ==========================================
# 🚀 第二部分：对外提供的预测接口 (每次用户请求都会执行)
# ==========================================
def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    BERT 核心推理函数
    :param request_text: 可以是一句话(str)，也可以是多句话的列表(List[str])
    :return: 对应的分类结果
    """
    # 统一把输入变成 List 格式，方便批量处理
    is_single = isinstance(request_text, str)
    texts = [request_text] if is_single else request_text

    # ---------------------------------------------------------
    # 步骤 1：汉字 -> 张量 (Tensor)
    # ---------------------------------------------------------
    # return_tensors="pt" 意思是直接返回 PyTorch 能看懂的 Tensor 矩阵！
    encoded_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    # 把输入数据也搬到 GPU/CPU 上，要跟模型在同一个设备
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # ---------------------------------------------------------
    # 步骤 2：张量 -> Logits (打分)
    # ---------------------------------------------------------
    # torch.no_grad() 的作用：告诉 PyTorch "只正向预测，不用算梯度"，能省下一半的内存并提速！
    with torch.no_grad():
        # 把数据喂给模型
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 取出 12 个类别的原始打分 (形状类似: [batch_size, 12])
    logits = outputs.logits

    # ---------------------------------------------------------
    # 步骤 3：Logits -> 意图类别
    # ---------------------------------------------------------
    # 把张量从显卡里抽出来，转成普通的 NumPy 数组
    logits = logits.detach().cpu().numpy()

    # np.argmax 找最大值：看看哪一列的分数最高，返回它的索引 (比如 0, 1, 5)
    pred_indices = np.argmax(logits, axis=1)

    # 根据索引，去配置表的 CATEGORY_NAMES 里找到真正的中文/英文名字
    results = [CATEGORY_NAMES[idx] for idx in pred_indices]

    # 如果进来的是一句话，出去也是一个字符串；如果是列表，出去也是列表
    return results[0] if is_single else results


# ==========================================
# 🧪 第三部分：本地测试
# ==========================================
if __name__ == "__main__":
    # 等你全量训练完，生成了 bert.pt 后，就可以运行这个文件测试了！
    test_sentences = [
        "明天深圳会不会下雨啊？",
        "帮我把客厅的空调打开，太热了",
        "我想听周杰伦的青花瓷",
        "最近有什么好看的动作电影吗"
    ]

    print("\n" + "=" * 60)
    print("开始 BERT 模型推理测试：")
    for sent in test_sentences:
        res = model_for_bert(sent)
        print(f"输入: {sent:25s} -> 预测意图: {res}")
    print("=" * 60 + "\n")