import numpy as np
import argparse
import os
from scipy.spatial import cKDTree

def calculate_similarity(pcd1: np.ndarray, pcd2: np.ndarray, threshold: float) -> tuple[float, float, float]:
    """
    使用精确率、召回率和F1-Score计算两个点云的相似度。

    Args:
        pcd1 (np.ndarray): 第一个点云 (通常视为参考或真值)。
        pcd2 (np.ndarray): 第二个点云 (通常视为生成或预测的结果)。
        threshold (float): 判断点是否对应的距离阈值（米）。

    Returns:
        A tuple containing:
        - precision (float): 精确率 (%)
        - recall (float): 召回率 (%)
        - f_score (float): F1-Score
    """
    # 创建高效的K-D树用于最近邻搜索
    pcd1_tree = cKDTree(pcd1)
    
    # --- 计算精确率 ---
    # 查询 pcd2 中的每个点在 pcd1 中的最近邻距离
    distances_p2_to_p1, _ = pcd1_tree.query(pcd2, k=1)
    # 计算距离小于阈值的点的比例
    precision = np.mean(distances_p2_to_p1 < threshold)

    # --- 计算召回率 ---
    pcd2_tree = cKDTree(pcd2)
    # 查询 pcd1 中的每个点在 pcd2 中的最近邻距离
    distances_p1_to_p2, _ = pcd2_tree.query(pcd1, k=1)
    # 计算距离小于阈值的点的比例
    recall = np.mean(distances_p1_to_p2 < threshold)

    # --- 计算 F1-Score ---
    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    return precision * 100, recall * 100, f_score

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="计算并比较两个NPY格式点云的点数和相似度。")
    parser.add_argument("file1", type=str, help="第一个点云文件的路径 (.npy)")
    parser.add_argument("file2", type=str, help="第二个点云文件的路径 (.npy)")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.2, 
        help="用于计算相似度的距离阈值（单位：米），默认为 0.2"
    )

    args = parser.parse_args()

    # --- 检查文件是否存在 ---
    if not os.path.exists(args.file1):
        print(f"错误: 文件未找到 -> {args.file1}")
        return
    if not os.path.exists(args.file2):
        print(f"错误: 文件未找到 -> {args.file2}")
        return

    # --- 加载点云 ---
    try:
        pcd1 = np.load(args.file1)
        pcd2 = np.load(args.file2)
        # 确保是 N x 3 的形状
        if pcd1.ndim != 2 or pcd1.shape[1] != 3:
            print(f"错误: {os.path.basename(args.file1)} 的形状不是 (N, 3)，而是 {pcd1.shape}")
            return
        if pcd2.ndim != 2 or pcd2.shape[1] != 3:
            print(f"错误: {os.path.basename(args.file2)} 的形状不是 (N, 3)，而是 {pcd2.shape}")
            return
    except Exception as e:
        print(f"加载NPY文件时出错: {e}")
        return

    # --- 计算并打印点数 ---
    print("-" * 50)
    print("点云点数统计:")
    print(f"  - 文件 1 ({os.path.basename(args.file1)}): {len(pcd1)} 个点")
    print(f"  - 文件 2 ({os.path.basename(args.file2)}): {len(pcd2)} 个点")
    print("-" * 50)

    # --- 计算并打印相似度 ---
    precision, recall, f_score = calculate_similarity(pcd1, pcd2, args.threshold)

    print(f"点云相似度评估 (阈值 = {args.threshold} 米):")
    print(f"  - 精确率 (Precision): {precision:.2f}%")
    print(f"    (解释: '{os.path.basename(args.file2)}' 中有多少比例的点，可以在 '{os.path.basename(args.file1)}' 中找到距离小于{args.threshold}米的对应点)")
    print(f"  - 召回率 (Recall):    {recall:.2f}%")
    print(f"    (解释: '{os.path.basename(args.file1)}' 中有多少比例的点，可以在 '{os.path.basename(args.file2)}' 中找到距离小于{args.threshold}米的对应点)")
    print(f"  - F1-Score:         {f_score:.4f}")
    print(f"    (解释: 精确率和召回率的综合评估分数，越接近1.0表示两个点云越相似)")
    print("-" * 50)


if __name__ == "__main__":
    main()
