# 文件名: fuzzy_deduplicate_local.py
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

class FuzzyDeduplicator:
    def __init__(self, num_perm=128, jaccard_threshold=0.85):
        """
        初始化去重器
        :param num_perm: MinHash的哈希函数数量，值越大精度越高
        :param jaccard_threshold: Jaccard相似度阈值，大于此值视为重复
        """
        self.num_perm = num_perm
        self.threshold = jaccard_threshold
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=num_perm)
        
    def _text_to_features(self, text, ngram_size=3):
        """
        将文本转换为特征集合（模拟Spark的HashingTF）
        :param text: 输入文本
        :param ngram_size: n-gram窗口大小
        """
        # 生成n-gram特征
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngrams.append(" ".join(words[i:i+ngram_size]))
        return set(ngrams)
    
    def _create_minhash(self, features):
        """
        从特征集合创建MinHash
        """
        m = MinHash(num_perm=self.num_perm)
        for f in features:
            m.update(f.encode('utf-8'))
        return m
    
    def deduplicate(self, documents):
        """
        执行模糊去重
        :param documents: 文档列表，格式为[(doc_id, text), ...]
        :return: 去重后的文档列表
        """
        # 第一阶段：构建LSH索引
        print("Building LSH index...")
        minhashes = {}
        for doc_id, text in tqdm(documents):
            features = self._text_to_features(text)
            minhash = self._create_minhash(features)
            self.lsh.insert(doc_id, minhash)
            minhashes[doc_id] = minhash
        
        # 第二阶段：查找重复项
        print("Finding duplicates...")
        duplicate_groups = {}
        for doc_id, minhash in tqdm(minhashes.items()):
            # 查找相似文档
            duplicates = self.lsh.query(minhash)
            # 排除自身
            duplicates = [d for d in duplicates if d != doc_id]
            if duplicates:
                # 将相似文档合并到同一组
                group_id = min([doc_id] + duplicates)
                for d in [doc_id] + duplicates:
                    if d not in duplicate_groups:
                        duplicate_groups[d] = group_id
        
        # 第三阶段：保留每组最小ID的文档
        print("Filtering duplicates...")
        seen_groups = set()
        dedup_documents = []
        for doc_id, text in documents:
            if doc_id in duplicate_groups:
                group_id = duplicate_groups[doc_id]
                if group_id not in seen_groups:
                    seen_groups.add(group_id)
                    dedup_documents.append((doc_id, text))
            else:
                dedup_documents.append((doc_id, text))
        
        return dedup_documents

if __name__ == "__main__":
    # 示例用法
    documents = [
        (0, "Apple is a leading tech company."),
        (1, "Apple Inc. is a top technology firm."),  # 与0相似
        (2, "Banana is a tropical fruit."),
        (3, "Bananas are grown in warm climates."),   # 与2相似
        (4, "Microsoft develops operating systems."),
    ]
    
    print("Original documents:")
    for doc in documents:
        print(f"ID {doc[0]}: {doc[1]}")
    
    # 初始化去重器（参数与Spark示例对应）
    deduplicator = FuzzyDeduplicator(
        num_perm=10,        # 对应numHashTables=10
        jaccard_threshold=0.85
    )
    
    # 执行去重
    dedup_docs = deduplicator.deduplicate(documents)
    
    print("\nDeduplicated documents:")
    for doc in dedup_docs:
        print(f"ID {doc[0]}: {doc[1]}")
