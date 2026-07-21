import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import joblib

def load_data():
    data_path = 'model_input_6_plus_spatial_856.csv'
    df = pd.read_csv(data_path)
    return df

def load_model():
    model_path = 'xgboost_6_plus_rg_optimized_full_model.joblib'
    model = joblib.load(model_path)
    return model

def load_params():
    return None

def prepare_features(df):
    six_features = [
        'Conju-Max-Distance',
        'Wavelength (Exp nm)',
        'ET(30) (Solvent)',
        'PEOE-Charge-Max',
        'Atomic-LogP-Min',
        'Atomic-MR-Max'
    ]
    features = six_features + ['rg']
    X = df[features]
    y = df['values_ln']
    return X, y, features

def shap_analysis(X, model, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("\n=== SHAP全局特征重要性 ===")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'SHAP Importance (Mean Absolute)': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP Importance (Mean Absolute)', ascending=False)
    print(feature_importance)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSHAP Summary Plot已保存为: shap_summary_plot.png")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar", show=False)
    plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP Bar Plot已保存为: shap_bar_plot.png")
    
    for i, feature in enumerate(features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(i, shap_values, X, feature_names=features, show=False)
        plt.savefig(f'shap_dependence_{feature}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print(f"SHAP Dependence Plots已保存 (共{len(features)}个)")
    
    print("\n=== SHAP特征贡献详细分析 ===")
    for i, feature in enumerate(features):
        mean_shap = np.mean(shap_values[:, i])
        std_shap = np.std(shap_values[:, i])
        max_shap = np.max(shap_values[:, i])
        min_shap = np.min(shap_values[:, i])
        print(f"\n{feature}:")
        print(f"  平均SHAP值: {mean_shap:.4f}")
        print(f"  SHAP值标准差: {std_shap:.4f}")
        print(f"  SHAP值范围: [{min_shap:.4f}, {max_shap:.4f}]")
        print(f"  平均绝对值贡献: {np.mean(np.abs(shap_values[:, i])):.4f}")
    
    return shap_values, explainer

def main():
    print("=" * 70)
    print("SHAP分析: 6特征 + rg + 超参数优化体系")
    print("=" * 70)
    
    print("\n1. 加载数据...")
    df = load_data()
    print(f"数据样本数: {len(df)}")
    print(f"特征列数: {len(df.columns)}")
    
    print("\n2. 加载优化后的模型...")
    model = load_model()
    
    print("\n3. 加载超参数...")
    params = load_params()
    print(f"最佳CV MSE (ln): {params['best_cv_mse_ln']:.6f}")
    print("最佳超参数:")
    for key, value in params['best_params'].items():
        print(f"  {key}: {value}")
    
    print("\n4. 准备特征 (6特征 + rg)...")
    X, y, features = prepare_features(df)
    print(f"特征列表: {features}")
    print(f"特征形状: {X.shape}")
    
    print("\n5. 执行SHAP分析...")
    shap_values, explainer = shap_analysis(X, model, features)
    
    print("\n" + "=" * 70)
    print("SHAP分析完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - shap_summary_plot.png: SHAP汇总散点图")
    print("  - shap_bar_plot.png: SHAP特征重要性条形图")
    print("  - shap_dependence_*.png: 各特征的SHAP依赖图")

if __name__ == "__main__":
    main()