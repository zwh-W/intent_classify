# train_bert.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 导入配置和日志
from config import DATA_PATH, BASE_DIR, CATEGORY_NAMES
from logger import logger

# ==========================================
# 📍 核心路径配置 (使用你本地下载好的模型)
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'models', 'AI-ModelScope', 'bert-base-chinese')
BERT_SAVE_DIR = os.path.join(BASE_DIR, "assets", "weights", "bert")
BERT_MODEL_PT = os.path.join(BASE_DIR, "assets", "weights", "bert.pt")


def compute_metrics(eval_pred):
    """
    【评估函数】
    除了准确率(Accuracy)，我们在工业界还必须看 F1 分数，
    防止模型只会猜数据最多的那个类别（样本不均衡问题）。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    # macro F1 能综合评估模型在各个类别上的平均表现
    f1 = f1_score(labels, predictions, average='macro')

    return {'accuracy': acc, 'f1_macro': f1}


def main():
    logger.info("=" * 60)
    logger.info("🚀 启动 BERT 模型微调训练 (精调版)")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ 找不到本地模型路径: {MODEL_PATH}")
        return

    # ---------------------------------------------------------
    # 步骤 1：读取数据 & 【修复标签映射问题】
    # ---------------------------------------------------------
    logger.info("[1/5] 读取数据与构建标签映射...")
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text', 'label'])

    # ⚠️ 核心修复：强制使用 config.py 中的 CATEGORY_NAMES 顺序，绝不依赖字母排序！
    # 建立 字符串 -> 数字 的映射字典：{'Travel-Query': 0, 'Music-Play': 1, ...}
    label2id = {name: idx for idx, name in enumerate(CATEGORY_NAMES)}

    # 把表格里的文字标签，替换成我们字典里的数字标签
    # 如果遇到不在 CATEGORY_NAMES 里的脏数据，直接丢弃（dropna）
    df['label_id'] = df['label'].map(label2id)
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)

    # 为了快速测试跑通，这里只取前 500 条。
    # 当你测试没问题了，把这里的 [:500] 删掉，就能训练全量数据了！
    texts = df['text'].values.tolist()
    labels_num = df['label_id'].values.tolist()
    num_classes = len(CATEGORY_NAMES)

    logger.info(f"本次训练加载了 {len(texts)} 条有效数据，共 {num_classes} 个意图类别。")

    # ---------------------------------------------------------
    # 步骤 2：数据集划分与 Tokenizer 编码
    # ---------------------------------------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels_num, test_size=0.2, random_state=42, stratify=labels_num
    )

    logger.info("[2/5] 加载 BERT 分词器，进行文本转张量(Tensor)...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    # 这一步把汉字变成了 input_ids 和 attention_mask
    train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

    # 封装成 HuggingFace 专用的 Dataset 格式
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
    # 步骤 3：加载带有分类头的 BERT 模型
    # ---------------------------------------------------------
    logger.info("[3/5] 初始化预训练模型 (BertForSequenceClassification)...")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=num_classes  # 告诉模型我们顶层外接一个 12 分类的线性层
    )

    # ---------------------------------------------------------
    # 步骤 4：定义训练参数并启动 Trainer
    # ---------------------------------------------------------
    logger.info("[4/5] 设置训练参数，开始炼丹...")
    training_args = TrainingArguments(
        output_dir=BERT_SAVE_DIR,
        num_train_epochs=3,  # 训练 4 轮
        per_device_train_batch_size=8,  # 如果你的显卡显存不够，这里改成 4
        per_device_eval_batch_size=16,
        eval_strategy="epoch",  # 每个 epoch 评估一次
        save_strategy="epoch",  # 每个 epoch 保存一次中间状态
        load_best_model_at_end=True,  # 【关键】训练结束后，自动把在验证集上表现最好的权重加载回来
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

    # 打印测试集上的最终成绩
    eval_results = trainer.evaluate()
    logger.info(f"🏆 最佳模型在测试集上的表现: {eval_results}")

    # ---------------------------------------------------------
    # 步骤 5：保存最纯粹的模型权重 (给预测代码用)
    # ---------------------------------------------------------
    logger.info("[5/5] 训练完成，剥离出核心状态字典并保存为 bert.pt ...")
    os.makedirs(os.path.dirname(BERT_MODEL_PT), exist_ok=True)
    torch.save(trainer.model.state_dict(), BERT_MODEL_PT)

    logger.info(f"✅ BERT 权重已成功保存至: {BERT_MODEL_PT}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()