import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

def load_data():
    data_path = r'C:\Users\HUAWEI\Desktop\5(1)\model_input_6_plus_spatial_856.csv'
    df = pd.read_csv(data_path)
    return df

def load_model():
    model_path = r'C:\Users\HUAWEI\Desktop\5(1)\xgboost_6_plus_rg_optimized_full_model.joblib'
    model = joblib.load(model_path)
    return model

def show_shap_matrix_details(shap_values, features, df):
    print("=" * 80)
    print("SHAP值矩阵内部细节")
    print("=" * 80)
    
    print(f"\n1. SHAP值矩阵形状: {shap_values.shape}")
    print(f"   - 样本数: {shap_values.shape[0]}")
    print(f"   - 特征数: {shap_values.shape[1]}")
    
    print("\n2. SHAP值矩阵前5行（每个样本的7个特征SHAP值）:")
    shap_df = pd.DataFrame(shap_values, columns=features)
    print(shap_df.head().to_string())
    
    print("\n3. 各特征SHAP值统计信息:")
    print(shap_df.describe().to_string())
    
    return shap_df

def show_sample_shap_decomposition(sample_idx, shap_values, features, df, explainer, model):
    print("\n" + "=" * 80)
    print(f"样本 {sample_idx+1} 的SHAP分解过程")
    print("=" * 80)
    
    sample_shap = shap_values[sample_idx]
    sample_data = df.iloc[sample_idx]
    prediction = model.predict(df[features].iloc[[sample_idx]])[0]
    baseline = explainer.expected_value
    
    print(f"\n基本信息:")
    print(f"   基线值 (所有样本平均预测): {baseline:.6f}")
    print(f"   该样本预测值: {prediction:.6f}")
    print(f"   预测偏差: {prediction - baseline:.6f}")
    
    print(f"\n样本 {sample_idx+1} 的原始特征值:")
    for feat in features:
        print(f"   {feat}: {sample_data[feat]:.4f}")
    
    print(f"\nSHAP分解过程:")
    print(f"   步骤0: 基线值 = {baseline:.6f}")
    
    current_value = baseline
    for i, feat in enumerate(features):
        current_value += sample_shap[i]
        sign = "+" if sample_shap[i] >= 0 else ""
        print(f"   步骤{i+1}: {sign}{sample_shap[i]:.6f} ({feat}) -> {current_value:.6f}")
    
    print(f"\n验证:")
    print(f"   Σ(SHAP值) = {sample_shap.sum():.6f}")
    print(f"   基线值 + Σ(SHAP值) = {baseline + sample_shap.sum():.6f}")
    print(f"   模型预测值 = {prediction:.6f}")
    print(f"   误差 = {abs(prediction - (baseline + sample_shap.sum())):.10f}")

def show_tree_structure(model):
    print("\n" + "=" * 80)
    print("XGBoost模型树结构信息")
    print("=" * 80)
    
    booster = model.get_booster()
    trees = booster.get_dump(with_stats=True)
    
    print(f"\n1. 模型基本信息:")
    print(f"   - 树数量 (n_estimators): {len(trees)}")
    print(f"   - 目标函数: {booster.attr('objective')}")
    
    print(f"\n2. 第1棵树的结构（前10个节点）:")
    tree_lines = trees[0].split('\n')[:10]
    for line in tree_lines:
        if line.strip():
            print(f"   {line}")
    
    print(f"\n3. 第2棵树的结构（前10个节点）:")
    tree_lines = trees[1].split('\n')[:10]
    for line in tree_lines:
        if line.strip():
            print(f"   {line}")
    
    print(f"\n4. TreeSHAP算法说明:")
    print("   - 遍历每棵树的决策路径")
    print("   - 在每个节点计算特征的边际贡献")
    print("   - 将所有树的贡献累加得到最终SHAP值")
    print("   - 复杂度: O(T * L * D^2)")
    print("     * T = 树数量 (517)")
    print("     * L = 树深度 (8)")
    print("     * D = 特征维度 (7)")

def generate_waterfall_plots(shap_values, features, df, explainer, sample_indices=[0, 100, 200]):
    print("\n" + "=" * 80)
    print("生成SHAP瀑布图")
    print("=" * 80)
    
    for idx in sample_indices:
        print(f"\n生成样本 {idx+1} 的瀑布图...")
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=df[features].iloc[idx],
            feature_names=features
        ), show=False)
        
        plt.title(f'SHAP Waterfall Plot - Sample {idx+1}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'shap_waterfall_sample_{idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   已保存: shap_waterfall_sample_{idx+1}.png")

def generate_summary_table(shap_values, features, df, explainer, model):
    print("\n" + "=" * 80)
    print("SHAP值汇总表")
    print("=" * 80)
    
    predictions = model.predict(df[features])
    shap_reconstruction = shap_values.sum(axis=1) + explainer.expected_value
    
    summary_df = pd.DataFrame({
        '样本索引': range(len(df)),
        '基线值': explainer.expected_value,
        'Conju-Max-Distance_SHAP': shap_values[:, 0],
        'Wavelength_SHAP': shap_values[:, 1],
        'ET(30)_SHAP': shap_values[:, 2],
        'PEOE-Charge_SHAP': shap_values[:, 3],
        'Atomic-LogP_SHAP': shap_values[:, 4],
        'Atomic-MR_SHAP': shap_values[:, 5],
        'rg_SHAP': shap_values[:, 6],
        'SHAP总和': shap_values.sum(axis=1),
        '重建预测值': shap_reconstruction,
        '模型预测值': predictions,
        '误差': np.abs(predictions - shap_reconstruction)
    })
    
    print("\n前10行数据:")
    print(summary_df.head(10).to_string())
    
    summary_df.to_csv('shap_values_full.csv', index=False)
    print("\n完整数据已保存到: shap_values_full.csv")
    
    return summary_df

def main():
    print("=" * 80)
    print("SHAP值内部细节分析")
    print("=" * 80)
    
    print("\n1. 加载数据...")
    df = load_data()
    
    features = ['Conju-Max-Distance', 'Wavelength (Exp nm)', 'ET(30) (Solvent)',
                'PEOE-Charge-Max', 'Atomic-LogP-Min', 'Atomic-MR-Max', 'rg']
    
    print("\n2. 加载模型...")
    model = load_model()
    
    print("\n3. 计算SHAP值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df[features])
    
    print("\n4. 展示SHAP值矩阵细节...")
    shap_df = show_shap_matrix_details(shap_values, features, df)
    
    print("\n5. 展示样本SHAP分解过程...")
    show_sample_shap_decomposition(0, shap_values, features, df, explainer, model)
    show_sample_shap_decomposition(100, shap_values, features, df, explainer, model)
    
    print("\n6. 展示模型树结构...")
    show_tree_structure(model)
    
    print("\n7. 生成SHAP瀑布图...")
    generate_waterfall_plots(shap_values, features, df, explainer)
    
    print("\n8. 生成完整SHAP值汇总表...")
    summary_df = generate_summary_table(shap_values, features, df, explainer, model)
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - shap_waterfall_sample_1.png")
    print("  - shap_waterfall_sample_101.png")
    print("  - shap_waterfall_sample_201.png")
    print("  - shap_values_full.csv")

if __name__ == "__main__":
    main()