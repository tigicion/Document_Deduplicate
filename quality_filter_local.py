# 文件名: quality_filter_local.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import re

# 示例数据格式（可替换为真实数据）
# 假设有以下CSV文件：
# - high_quality_data.csv 含 "text" 列
# - common_crawl_sample.csv 含 "text" 列

# 加载并预处理数据（替换为实际文件路径）
def load_data(pos_path, neg_path):
    # 加载正样本
    pos_df = pd.read_csv(pos_path)
    pos_df['label'] = 1
    
    # 加载负样本
    neg_df = pd.read_csv(neg_path)
    neg_df['label'] = 0
    
    # 下采样负样本（保持正样本的10倍量）
    sample_size = min(10 * len(pos_df), len(neg_df))
    neg_df = neg_df.sample(sample_size, random_state=42)
    
    # 合并数据
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    
    # 简单文本清洗
    df['text'] = df['text'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))  # 去HTML标签
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x)))  # 去除非字母字符
    
    return df

# 加载训练数据
train_df = load_data("high_quality_data.csv", "common_crawl_sample.csv")

# 特征工程（替代Spark的Tokenizer+HashingTF）
vectorizer = HashingVectorizer(
    n_features=2**20,  # 与Spark的numFeatures=2**20对应
    alternate_sign=False,  # 保持非负值以匹配HashingTF行为
    norm=None
)

X = vectorizer.fit_transform(train_df['text'])
y = train_df['label'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练逻辑回归模型
lr = LogisticRegression(
    max_iter=100,
    C=0.01,  # 对应regParam=0.01 (C=1/regParam)
    solver='saga',  # 支持L1/L2正则
    random_state=42
)
lr.fit(X_train, y_train)

# 评估模型
y_pred_proba = lr.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Validation AUC: {auc:.4f}")

# 加载待过滤数据（Common Crawl）
# 假设有 common_crawl.csv 含 "text" 列
cc_df = pd.read_csv("common_crawl.csv")
cc_df['text'] = cc_df['text'].apply(lambda x: re.sub(r'<[^>]+>', '', str(x)))
cc_df['text'] = cc_df['text'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x)))

# 生成预测得分
X_cc = vectorizer.transform(cc_df['text'])
cc_scores = lr.predict_proba(X_cc)[:, 1]

# 帕累托重采样
alpha = 9

def pareto_filter(score):
    return np.random.pareto(alpha) > (1 - score)

# 应用过滤
filter_mask = [pareto_filter(score) for score in cc_scores]
filtered_df = cc_df[filter_mask]

# 保存结果
filtered_df.to_csv("filtered_common_crawl.csv", index=False)

print(f"原始数据量: {len(cc_df)}")
print(f"过滤后数据量: {len(filtered_df)}")
print(f"过滤比例: {1 - len(filtered_df)/len(cc_df):.2%}")

# 示例数据生成（如果无真实数据可取消注释以下代码创建示例）
"""
if __name__ == "__main__":
    # 生成示例数据
    import numpy as np
    np.random.seed(42)
    
    # 生成高质量数据（正样本）
    high_quality = pd.DataFrame({
        'text': [f"well-written document {i} about machine learning" for i in range(1000)]
    })
    high_quality.to_csv("high_quality_data.csv", index=False)
    
    # 生成CommonCrawl样本（负样本）
    common_crawl = pd.DataFrame({
        'text': [f"random web content {i} with <html> tags" + (" and noise" * (i%10)) for i in range(10000)]
    })
    common_crawl.to_csv("common_crawl_sample.csv", index=False)
    
    # 生成待过滤数据
    test_data = pd.DataFrame({
        'text': [f"test document {i} with mixed quality" for i in range(5000)]
    })
    test_data.to_csv("common_crawl.csv", index=False)
"""
