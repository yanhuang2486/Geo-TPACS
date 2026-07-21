import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib

def verify_shap_calculation():
    print("=" * 70)
    print("SHAP计算准确性验证")
    print("=" * 70)
    
    data_path = 'model_input_6_plus_spatial_856.csv'
    model_path = 'xgboost_6_plus_rg_optimized_full_model.joblib'
    
    print("\n1. 加载数据...")
    df = pd.read_csv(data_path)
    
    features = ['Conju-Max-Distance', 'Wavelength (Exp nm)', 'ET(30) (Solvent)', 
                'PEOE-Charge-Max', 'Atomic-LogP-Min', 'Atomic-MR-Max', 'rg']
    X = df[features]
    y = df['values_ln']
    
    print(f"   样本数: {len(df)}")
    print(f"   特征数: {len(features)}")
    
    print("\n2. 加载模型...")
    model = joblib.load(model_path)
    
    print("\n3. 计算SHAP值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print(f"   基线值 (expected_value): {explainer.expected_value:.6f}")
    print(f"   SHAP值形状: {shap_values.shape}")
    
    print("\n4. 验证核心公式: Σ(SHAP值) + 基线值 = 预测值")
    predictions = model.predict(X)
    shap_reconstruction = shap_values.sum(axis=1) + explainer.expected_value
    
    max_error = np.abs(predictions - shap_reconstruction).max()
    mean_error = np.abs(predictions - shap_reconstruction).mean()
    std_error = np.abs(predictions - shap_reconstruction).std()
    
    print(f"   最大误差: {max_error:.10f}")
    print(f"   平均误差: {mean_error:.10f}")
    print(f"   误差标准差: {std_error:.10f}")
    
    if max_error < 1e-4:
        print("   \u2705 验证通过: SHAP值计算准确 (误差在浮点精度范围内)")
    else:
        print("   \u274C 验证失败: SHAP值计算存在误差")
    
    print("\n5. 验证前5个样本:")
    for i in range(5):
        print(f"   样本 {i+1}:")
        print(f"      模型预测: {predictions[i]:.6f}")
        print(f"      SHAP重建: {shap_reconstruction[i]:.6f}")
        print(f"      差值: {predictions[i] - shap_reconstruction[i]:.10f}")
    
    print("\n6. 验证SHAP值分布:")
    print(f"   SHAP值范围: [{shap_values.min():.4f}, {shap_values.max():.4f}]")
    print(f"   SHAP值均值: {shap_values.mean():.6f}")
    print(f"   基线值 ≈ 训练集平均预测值: {explainer.expected_value:.6f} vs {predictions.mean():.6f}")
    
    print("\n" + "=" * 70)
    print("验证完成!")
    print("=" * 70)
    
    return {
        'max_error': max_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'baseline': explainer.expected_value,
        'shap_shape': shap_values.shape
    }

if __name__ == "__main__":
    results = verify_shap_calculation()