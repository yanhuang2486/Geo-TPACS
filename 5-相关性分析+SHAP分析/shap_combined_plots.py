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

def prepare_features(df):
    features = [
        'Conju-Max-Distance',
        'Wavelength (Exp nm)',
        'ET(30) (Solvent)',
        'PEOE-Charge-Max',
        'Atomic-LogP-Min',
        'Atomic-MR-Max',
        'rg'
    ]
    X = df[features]
    y = df['values_ln']
    return X, y, features

def plot_feature_importance(shap_values, features):
    importance = np.abs(shap_values).mean(axis=0)
    df_importance = pd.DataFrame({
        'Feature': features,
        'Mean Absolute SHAP': importance
    }).sort_values(by='Mean Absolute SHAP', ascending=True)
    
    norm = plt.Normalize(df_importance['Mean Absolute SHAP'].min(), df_importance['Mean Absolute SHAP'].max())
    cmap = matplotlib.colormaps['coolwarm']
    
    colors = cmap(norm(df_importance['Mean Absolute SHAP']))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_importance['Feature'], df_importance['Mean Absolute SHAP'], color=colors)
    
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Mean Absolute SHAP Value')
    
    for bar, value in zip(bars, df_importance['Mean Absolute SHAP']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{value:.4f}', va='center', fontsize=10)
    
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance Ranking (Mean |SHAP|)', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_importance_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("特征重要性排名图已保存为: shap_importance_ranking.png")
    
    print("\n=== 特征重要性排名 ===")
    print(df_importance[::-1].to_string(index=False))
    
    return df_importance

def plot_combined_dependence(shap_values, X, features):
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))
    
    for i, (feature, ax) in enumerate(zip(features, axes.flat)):
        shap.dependence_plot(
            i, shap_values, X, feature_names=features,
            ax=ax, show=False, interaction_index=None
        )
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('SHAP Value', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
    
    if len(features) < len(axes.flat):
        for j in range(len(features), len(axes.flat)):
            axes.flat[j].axis('off')
    
    plt.tight_layout()
    plt.savefig('shap_dependence_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n合并SHAP依赖图已保存为: shap_dependence_combined.png")

def main():
    print("=" * 70)
    print("生成SHAP特征重要性排名图和合并依赖图")
    print("=" * 70)
    
    print("\n1. 加载数据...")
    df = load_data()
    
    print("\n2. 加载模型...")
    model = load_model()
    
    print("\n3. 准备特征...")
    X, y, features = prepare_features(df)
    
    print("\n4. 计算SHAP值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("\n5. 生成特征重要性排名图...")
    plot_feature_importance(shap_values, features)
    
    print("\n6. 生成合并SHAP依赖图...")
    plot_combined_dependence(shap_values, X, features)
    
    print("\n" + "=" * 70)
    print("图片生成完成!")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - shap_importance_ranking.png: 特征重要性排名图")
    print("  - shap_dependence_combined.png: 7个特征的合并SHAP依赖图")

if __name__ == "__main__":
    main()