import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---- 全局字体设置（增大字号） ----
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 增大各类字体
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def main():
    # 1. 读取数据
    df = pd.read_csv('output.csv', encoding='utf-8-sig')
    print(f"原始数据行数: {len(df)}")

    # 2. 数据清洗：删除含缺失值的行
    df_clean = df.dropna()
    print(f"清洗后行数: {len(df_clean)}")

    if df_clean.empty:
        print("清洗后无数据，无法绘图。")
        return

    # 保存清洗后的数据
    clean_file = 'cleaned_output.csv'
    df_clean.to_csv(clean_file, index=False, encoding='utf-8-sig')
    print(f"清洗后数据已保存为 {clean_file}")

    # 定义特征列（包含所有可能需要的特征）
    all_feature_cols = ['rg', 'asphericity', 'eccentricity', 
                        'num_chiral_centers', 'num_cis_trans', 
                        'rg_mw_ratio', 'molecular_weight', 'rigid_bond_ratio']
    # 排除波长（wavelength）用于热力图和柱状图
    feature_cols_no_wavelength = [col for col in all_feature_cols if col in df_clean.columns]
    # 用于分布图的六个特征（确保存在）
    dist_features = ['rg', 'asphericity', 'eccentricity', 
                     'rg_mw_ratio', 'molecular_weight', 'rigid_bond_ratio']
    # 检查哪些列存在
    missing_dist = [col for col in dist_features if col not in df_clean.columns]
    if missing_dist:
        print(f"警告：分布图中以下列不存在: {missing_dist}")
        # 仅使用存在的列
        dist_features = [col for col in dist_features if col in df_clean.columns]
        if not dist_features:
            print("没有可用的分布特征，退出。")
            return

    target_col = 'TPACS'

    # --- 图1：六个特征的分布图（2×3） ---
    n_dist = len(dist_features)
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    # 如果特征数不足6个，多余的子图隐藏
    for idx, ax in enumerate(axes.flat):
        if idx < n_dist:
            col = dist_features[idx]
            sns.histplot(df_clean[col], kde=True, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontsize=16)   # 可单独指定
            ax.set_xlabel(col, fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分布图已保存为 distribution.png")

    # --- 图2：除波长外特征间的相关性热力图 ---
    if len(feature_cols_no_wavelength) < 2:
        print("特征数量不足，无法绘制热力图。")
    else:
        corr_matrix = df_clean[feature_cols_no_wavelength].corr(method='spearman')
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=0.5,
                    annot_kws={'size': 12})  # 增大热力格中数字
        plt.title('Spearman Correlation among Features (excluding wavelength)', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("特征相关性热力图已保存为 correlation_heatmap.png")

    # --- 图3：除波长外各特征与TPACS的Spearman相关系数柱状图 ---
    if len(feature_cols_no_wavelength) == 0:
        print("没有特征可用于计算与TPACS的相关性。")
    else:
        spearman_corr = {}
        for col in feature_cols_no_wavelength:
            corr_val = df_clean[[col, target_col]].corr(method='spearman').iloc[0,1]
            spearman_corr[col] = corr_val
        # 排序
        sorted_items = sorted(spearman_corr.items(), key=lambda x: x[1])
        features_sorted = [item[0] for item in sorted_items]
        corrs_sorted = [item[1] for item in sorted_items]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(features_sorted, corrs_sorted, color='steelblue')
        # 柱上数值标签，字号设为12
        for bar, val in zip(bars, corrs_sorted):
            plt.text(val + 0.01 * (1 if val >= 0 else -1), 
                     bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=12)
        plt.xlabel('Spearman Correlation with TPACS', fontsize=14)
        plt.title('Spearman Correlation of Features (excluding wavelength) with TPACS', fontsize=16)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        plt.savefig('spearman_barplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Spearman相关系数柱状图已保存为 spearman_barplot.png")

        # 打印相关系数
        print("\n各特征（除波长外）与 TPACS 的 Spearman 相关系数:")
        for col, val in sorted_items:
            print(f"  {col}: {val:.4f}")

if __name__ == "__main__":
    main()