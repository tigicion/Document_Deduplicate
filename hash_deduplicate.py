from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors

# 假设已通过HashingTF生成特征列"features"
# dataset = ...  # 包含"features"列的DataFrame

# 初始化MinHashLSH模型
minhash = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=10  # 10个哈希表
)

# 训练模型（计算哈希签名）
model = minhash.fit(dataset)

# 对数据生成哈希签名
signed_data = model.transform(dataset)

# 查找相似文档对（阈值可调）
similarity_threshold = 0.85  # Jaccard相似度阈值
duplicate_pairs = model.approxSimilarityJoin(
    signed_data, signed_data, threshold=similarity_threshold
)

# 标记重复文档（假设保留每组中id最小的文档）
from pyspark.sql import Window
from pyspark.sql.functions import row_number

# 定义窗口函数按文档ID排序
window = Window.partitionBy("datasetA.id").orderBy("datasetB.id")

# 保留每组中第一个文档，其余标记为重复
deduplicated = duplicate_pairs.withColumn(
    "rank", row_number().over(window)
).filter("rank == 1").select("datasetA.*")

# 合并非重复文档
final_data = dataset.join(deduplicated, "id", "left_anti")
