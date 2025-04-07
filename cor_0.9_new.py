# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

# ------------------- 数据加载与预处理 -------------------
def load_and_preprocess(file_path, sep, index_col):
    """加载数据并预处理"""
    df = pd.read_csv(file_path, sep=sep, index_col=index_col)
    df = df[(df > 1).sum(axis=1) >= 5]  # 过滤低表达基因
    df = np.log2(df + 1)                # Log2转换
    return df[df.std(axis=1) > 0]        # 移除恒定基因

# 加载数据
df_lnc = load_and_preprocess('filtered_nllnc.csv', ',', 'Gene')
df_mrna = load_and_preprocess('filtered_nlgene.tsv', '\t', 'Gene')
df_lnc = df_lnc[df_mrna.columns]  # 对齐样本列

# ------------------- 计算相关系数（保留原始值） -------------------
# 转换为标准化矩阵
lnc_matrix = df_lnc.values.astype(np.float32)
mrna_matrix = df_mrna.values.astype(np.float32)
n_samples = lnc_matrix.shape[1]

# 标准化
lnc_centered = lnc_matrix - lnc_matrix.mean(axis=1, keepdims=True)
mrna_centered = mrna_matrix - mrna_matrix.mean(axis=1, keepdims=True)
lnc_std = lnc_centered / (lnc_centered.std(axis=1, ddof=1, keepdims=True) + 1e-8)
mrna_std = mrna_centered / (mrna_centered.std(axis=1, ddof=1, keepdims=True) + 1e-8)

# 分块计算
block_size = 500
n_lnc = lnc_std.shape[0]
corr_matrix = np.zeros((n_lnc, mrna_std.shape[0]), dtype=np.float32)

for start in tqdm(range(0, n_lnc, block_size), desc='计算进度'):
    end = min(start + block_size, n_lnc)
    block = lnc_std[start:end]
    corr_matrix[start:end] = np.dot(block, mrna_std.T) / (n_samples - 1)

# ------------------- 输出原始相关系数（包含正负） -------------------
# 筛选条件：绝对值 >0.9，但保留原始值
rows, cols = np.where(np.abs(corr_matrix) > 0.9)
result = pd.DataFrame({
    'lncRNA': df_lnc.index[rows],
    'mRNA': df_mrna.index[cols],
    'Pearson': corr_matrix[rows, cols]  # 直接使用原始值
})

# 保存结果
result.to_csv('pearson_raw_0.9.tsv', sep='\t', index=False)
print(f"保存完成！相关系数保留原始正负值，数据量：{len(result)}")
