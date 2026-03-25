# train_bert.py
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 导入你写好的配置
from config import DATA_PATH, BASE_DIR
from logger import logger

# ==========================================
# 📍 核心路径配置 (使用你的本地模型)
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'models', 'AI-ModelScope', 'bert-base-chinese')
BERT_SAVE_DIR = os.path.join(BASE_DIR, "assets", "weights", "bert")
BERT_MODEL_PT = os.path.join(BASE_DIR, "assets", "weights", "bert.pt")


def demo_bert_theory(tokenizer):
    """
    【原理解密】：让你亲眼看看 BERT 是怎么理解句子的
    """
    print("\n" + "=" * 50)
    print("🕵️‍♂️ BERT 原理小测试：亲眼看看 Tokenization")
    text = "帮我开一下空调"
    print(f"原始句子: {text}")

    # 1. 编码：把句子转成 BERT 认识的格式
    encoded = tokenizer(text)

    # 2. 解码：看看它到底切成了什么词
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    print(f"被 BERT 切分后的 Token: {tokens}")
    print(f"对应的数字 ID (input_ids): {encoded['input_ids']}")
    print("👉 看到没？句子开头自动加了 [CLS]，结尾加了 [SEP]！")
    print("=" * 50 + "\n")


def compute_metrics(eval_pred):
    """准确率计算函数"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}


def main():
    logger.info("=" * 60)
    logger.info("开始 BERT 模型的微调训练流程...")

    # 确保本地模型路径存在
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ 找不到本地模型路径: {MODEL_PATH}")
        logger.error("请检查文件是否下载完整，或者路径拼写是否正确！")
        return

    # ---------------------------------------------------------
    # 第 0 步：【原理解密】让你秒懂 BERT 分词器
    # ---------------------------------------------------------
    # 直接从本地路径加载分词器
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    demo_bert_theory(tokenizer)  # 运行上面的小测试函数

    # ---------------------------------------------------------
    # 第 1 步：读取数据并转换标签
    # ---------------------------------------------------------
    logger.info("[1/5] 读取数据...")
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text', 'label'])

    # 【MVP测试】：为了让你几分钟内跑完，我们先只取前 500 条数据！
    texts = df['text'].values[:500].tolist()
    labels_text = df['label'].values[:500]

    # 将文字标签转为 0,1,2... 数字
    lbl_encoder = LabelEncoder()
    labels_num = lbl_encoder.fit_transform(labels_text)
    num_classes = len(lbl_encoder.classes_)
    logger.info(f"本次训练使用 {len(texts)} 条数据，共 {num_classes} 个意图类别。")

    # ---------------------------------------------------------
    # 第 2 步：划分数据集并进行编码
    # ---------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels_num, test_size=0.2, random_state=42, stratify=labels_num
    )

    logger.info("[2/5] 使用 BERT Tokenizer 批量编码文本...")
    # padding=True 长度不够补0；truncation=True 长度超了截断；max_length=64 句子最长64个字
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train
    })
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': y_test
    })

    # ---------------------------------------------------------
    # 第 3 步：加载本地预训练模型
    # ---------------------------------------------------------
    logger.info("[3/5] 加载本地 BERT 模型结构...")
    # 从本地加载模型，告诉它我们要分多少个类
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=num_classes
    )

    # ---------------------------------------------------------
    # 第 4 步：配置 Trainer 并训练
    # ---------------------------------------------------------
    logger.info("[4/5] 启动 HuggingFace Trainer...")
    training_args = TrainingArguments(
        output_dir=BERT_SAVE_DIR,
        num_train_epochs=3,  # 跑 3 轮
        per_device_train_batch_size=8,  # 每批处理 8 个样本
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",  # 每轮结束考试一次
        save_strategy="epoch",  # 每轮结束存档一次
        load_best_model_at_end=True,  # 训练结束保留最好的一份
        logging_dir='./logs',
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # 打印最终考试成绩
    eval_results = trainer.evaluate()
    logger.info(f"🎉 验证集评估成绩: {eval_results}")

    # ---------------------------------------------------------
    # 第 5 步：保存只属于你的微调权重
    # ---------------------------------------------------------
    logger.info("[5/5] 保存最佳模型权重...")
    os.makedirs(os.path.dirname(BERT_MODEL_PT), exist_ok=True)
    # 把大脑里的参数抽出来，存成一个 bert.pt 文件，留给后面的接口推理用
    torch.save(trainer.model.state_dict(), BERT_MODEL_PT)

    logger.info(f"✅ BERT 权重已成功保存至: {BERT_MODEL_PT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()