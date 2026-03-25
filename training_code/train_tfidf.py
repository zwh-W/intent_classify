# train_tfidf.py
import os
import pandas as pd
import jieba
from joblib import dump

# Sklearn 工具包（机器学习三件套）
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# 导入我们自己的模块
from config import DATA_PATH, STOPWORDS_PATH, TFIDF_MODEL_PATH, CATEGORY_NAMES
from logger import logger  # 直接导入我们配置好的彩色 logger


def main():
    logger.info("=" * 60)
    logger.info("开始训练 TF-IDF + LinearSVC 模型")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # 第二步：读取数据
    # --------------------------------------------------------
    logger.info("[1/6] 正在读取数据...")

    if not os.path.exists(DATA_PATH):
        logger.error(f"找不到数据集：{DATA_PATH}")
        return
    # 重点 2：sep='\t' 是核心！告诉 Pandas 按制表符分割
    # header=None：文件第一行不是表头，是数据
    # names=['text', 'label']：手动给两列起名字
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['text', 'label'])
    logger.info(f"数据读取成功！共 {len(df)} 条。")

    # --------------------------------------------------------
    # 第三步：文本预处理
    # --------------------------------------------------------
    # --------------------------------------------------------
    # 第三步：文本预处理（修复版）
    # --------------------------------------------------------
    logger.info("[2/6] 正在进行文本预处理...")

    # 1. 加载停用词
    cn_stopwords = pd.read_csv(STOPWORDS_PATH, header=None)[0].values
    # 🔧 修复1：把 NumPy 数组转成 Python 的 set，查找速度更快且不会报错
    cn_stopwords = set(cn_stopwords)

    # 2. 定义预处理函数
    def preprocess(text):
        # 🔧 修复2：加上 if！这才是正确的列表推导式语法
        words = [w for w in jieba.lcut(text) if w not in cn_stopwords]
        return " ".join(words)

    # 3. apply 批量处理
    df['text_clean'] = df['text'].apply(preprocess)

    logger.info("文本预处理完成。")

    # --------------------------------------------------------
    # 第四步：划分数据集
    # --------------------------------------------------------
    logger.info("[3/6] 正在划分训练集和测试集...")

    # X 是特征（干净的文本），y 是标签（意图）
    X = df['text_clean']
    y = df['label']

    # 重点：train_test_split
    # test_size=0.2：20% 数据当测试集
    # random_state=42：随机种子，保证每次运行划分结果一样
    # stratify=y：保证训练集和测试集的标签比例一致（比如少数类不会在测试集里消失）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"训练集：{len(X_train)}，测试集：{len(X_test)}")

    # ... (接上面的代码)

    # --------------------------------------------------------
    # 第五步：特征提取 + 训练模型
    # --------------------------------------------------------
    logger.info("[4/6] 正在进行 TF-IDF 特征提取...")

    # 1. 初始化 TF-IDF 工具
    # max_features=10000：只保留最常用的 10000 个词，防止电脑卡死
    tfidf = TfidfVectorizer(max_features=10000)
    # 2. 转换训练集
    # fit_transform：先“学习”词表（fit），再转成向量（transform）
    X_train_vec = tfidf.fit_transform(X_train)

    # 3. 转换测试集
    # 注意：只能用 transform！绝对不能用 fit_transform！
    # 因为测试集是“考试卷”，你不能考前先看卷子（不能让模型从测试集学词表）
    X_test_vec = tfidf.transform(X_test)

    logger.info("[5/6] 正在训练 LinearSVC 模型...")

    # 4. 初始化模型
    # class_weight='balanced'：自动给少数类加权重，解决数据不平衡
    model = LinearSVC(class_weight='balanced', random_state=42)

    # 5. 训练（就这一行，sklearn 帮你搞定所有数学）
    model.fit(X_train_vec, y_train)

    logger.info("模型训练完成！")
    # --------------------------------------------------------
    # 第六步：评估与保存
    # --------------------------------------------------------
    logger.info("[6/6] 正在评估模型...")

    # 1. 预测
    y_pred = model.predict(X_test_vec)

    # 2. 计算准确率
    acc = accuracy_score(y_test, y_pred)
    logger.info(f" 模型测试集准确率：{acc:.4f}")

    # 3. 打印详细报告（重点看少数类的 Recall）
    logger.info("\n详细分类报告：\n" + classification_report(y_test, y_pred, target_names=CATEGORY_NAMES))

    # 4. 保存模型
    # 必须同时保存 tfidf（向量化工具）和 model（分类模型），缺一不可！
    logger.info("正在保存模型...")
    dump((tfidf, model), TFIDF_MODEL_PATH)

    logger.info(f" 模型已保存至：{TFIDF_MODEL_PATH}")
    logger.info("=" * 60)
    logger.info("训练全部结束！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
