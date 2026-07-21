import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_shap_detailed_intro():
    fig = plt.figure(figsize=(16, 14), dpi=200)
    
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    def draw_block(x, y, width, height, text_lines, fontsize=11, title=None, title_fontsize=14, bg_color='#f8f9fa', edge_color='#dee2e6'):
        rect = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.3",
            facecolor=bg_color,
            edgecolor=edge_color,
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        if title:
            ax.text(x + width/2, y + height - 2.5, title, 
                   fontsize=title_fontsize,
                   ha='center', va='center', color='#2c3e50')
        
        line_spacing = 0.65
        start_y = y + height - 6 if title else y + height - 2.5
        for i, line in enumerate(text_lines):
            ax.text(x + 2, start_y - i * line_spacing, line,
                   fontsize=fontsize, ha='left', va='center', color='#34495e')
    
    draw_block(
        x=5, y=85, width=90, height=12,
        title="SHAP (SHapley Additive exPlanations) 方法简介",
        text_lines=[
            "SHAP 是基于合作博弈论中 Shapley 值的模型解释方法，将模型预测视为多个特征'合作'的结果，",
            "SHAP 值衡量每个特征在所有可能特征组合中的平均边际贡献"
        ],
        fontsize=12,
        title_fontsize=16
    )
    
    draw_block(
        x=5, y=70, width=45, height=13,
        title="核心公式",
        text_lines=[
            r"$\sum_{i=1}^{n} \phi_i(x) = f(x) - E[f(x)]$",
            "",
            "其中：",
            "  φ_i(x) = 特征 i 对样本 x 的 SHAP 值",
            "  f(x) = 模型对样本 x 的预测值",
            "  E[f(x)] = 所有样本的平均预测值（基线值）"
        ],
        fontsize=11,
        title_fontsize=14,
        bg_color='#e8f4fd',
        edge_color='#3498db'
    )
    
    draw_block(
        x=52, y=70, width=43, height=13,
        title="Shapley 值定义",
        text_lines=[
            r"$\phi_i = \sum_{S \subseteq N \setminus \{i\}}$",
            r"$\frac{|S|! \cdot (n-|S|-1)!}{n!}$",
            r"$\times [f(S \cup \{i\}) - f(S)]$",
            "",
            "解释：遍历所有不含特征 i 的子集 S，",
            "计算特征 i 加入 S 后的边际贡献，",
            "按子集大小加权平均"
        ],
        fontsize=10,
        title_fontsize=14,
        bg_color='#fef9e7',
        edge_color='#f39c12'
    )
    
    draw_block(
        x=5, y=52, width=90, height=16,
        title="TreeSHAP 算法细节",
        text_lines=[
            "本项目使用 TreeSHAP 算法（TreeExplainer），专门针对树模型（XGBoost、Random Forest等）优化",
            "",
            "算法特点：",
            "  1. 复杂度：O(TLD^2)，其中 T=树数量，L=树深度，D=特征维度",
            "  2. 原理：通过遍历树结构，递归计算每个特征在每个节点的边际贡献",
            "  3. 优化：利用树的结构特性，避免枚举所有 2^n 个子集，大幅提升计算效率",
            "  4. 精确性：对树模型可精确计算 SHAP 值，无需近似",
            "",
            "计算过程：从根节点到叶节点，追踪每个特征对预测路径的贡献"
        ],
        fontsize=11,
        title_fontsize=15,
        bg_color='#e8f8f5',
        edge_color='#1abc9c'
    )
    
    draw_block(
        x=5, y=32, width=45, height=18,
        title="当前模型关联",
        text_lines=[
            "本项目使用的模型：XGBoost 回归模型",
            "",
            "模型参数（超参数优化后）：",
            "  - n_estimators = 517（树数量）",
            "  - max_depth = 8（树深度）",
            "  - learning_rate = 0.0306",
            "  - reg_alpha = 1.42e-4（L1正则化）",
            "  - reg_lambda = 1.28e-4（L2正则化）",
            "",
            "SHAP 计算方式：",
            "  - 基线值 = 训练集平均预测值",
            "  - 每个样本的预测值 = 基线值 + Σ(各特征SHAP值)"
        ],
        fontsize=11,
        title_fontsize=14,
        bg_color='#f5eeff',
        edge_color='#9b59b6'
    )
    
    draw_block(
        x=52, y=32, width=21, height=18,
        title="SHAP 值含义",
        text_lines=[
            "SHAP 值为正：",
            "  该特征对分子的",
            "  TPACS 预测有",
            "  提升作用",
            "",
            "SHAP 值为负：",
            "  该特征对分子的",
            "  TPACS 预测有",
            "  抑制作用",
            "",
            "绝对值越大：",
            "  特征影响越强"
        ],
        fontsize=11,
        title_fontsize=14
    )
    
    draw_block(
        x=75, y=32, width=20, height=18,
        title="SHAP 优势",
        text_lines=[
            "全局层面：",
            "  评估特征整体",
            "  重要性",
            "  识别关键特征",
            "",
            "局部层面：",
            "  解释单个样本",
            "  预测逻辑",
            "  理解特征交互",
            "",
            "理论保证：",
            "  Shapley 值公理",
            "  一致性、效率性"
        ],
        fontsize=11,
        title_fontsize=14
    )
    
    ax.text(50, 8, "注：TPACS = Thermally Activated Phase Change Switch | 本图基于 6特征+rg+超参数优化的 XGBoost 模型",
           fontsize=10, ha='center', va='center', color='#95a5a6')
    
    plt.savefig('shap_algorithm_details.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("SHAP算法细节图已保存为: shap_algorithm_details.png")

if __name__ == "__main__":
    create_shap_detailed_intro()