# config.py
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据集路径
DATA_PATH = os.path.join(BASE_DIR, 'assets', 'dataset', 'dataset.csv')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'assets', 'dataset', 'baidu_stopwords.txt')

# 模型保存路径
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'weights', 'tfidf_ml.pkl')

# 意图类别列表（12个类别）
CATEGORY_NAMES = [
    'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
    'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
    'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play',
    'Other'
]