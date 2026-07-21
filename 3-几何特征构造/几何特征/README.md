# 分子特征计算方式说明报告

本报告概述了从分子结构（SMILES 和三维坐标）中提取的 9 个特征的定义、计算方法及物理/化学意义。所有计算均基于 `anti-rotation.py` 脚本的实现。

---

## 特征列表

| 特征名 | 缩写/符号 | 计算方式简述 | 数据来源 |
|--------|-----------|--------------|----------|
| **回旋半径** | `rg` | 根据分子三维坐标计算质量加权均方根距离 | `rdMolDescriptors.CalcRadiusOfGyration` |
| **非球形度** | `asphericity` | 描述分子形状偏离球体的程度，基于惯量张量 | `rdMolDescriptors.CalcAsphericity` |
| **离心率** | `eccentricity` | 描述分子形状的椭球拉伸程度 | `rdMolDescriptors.CalcEccentricity` |
| **手性中心数** | `num_chiral_centers` | SMILES 中四面体手性原子的数量（加氢后统计） | `rdMolDescriptors.CalcNumAtomStereoCenters` |
| **顺反异构数** | `num_cis_trans` | SMILES 中 E/Z 双键立体异构的数量 | 通过 `GetStereoInfo` 筛选 `STEREO_E` 类型 |
| **波长** | `wavelength` | 光谱测量的波长（nm），直接取自输入数据 | JSON 中的 `wavelength` 字段 |
| **rg/分子量比值** | `rg_mw_ratio` | 回旋半径除以分子量，反映分子紧凑程度与大小的关系 | `rg / MolWt` |
| **相对分子量** | `molecular_weight` | 分子量（g/mol），基于 SMILES 计算 | `Descriptors.MolWt` |
| **刚性键比例** | `rigid_bond_ratio` | 刚性键（双键、三键、环内单键、轴手性键）占非氢键总数的比例 | 自定义算法（见下文） |

---

## 各特征详细计算说明

### 1. 回旋半径 (Radius of Gyration, `rg`)
- **定义**：分子中所有原子到其质心的均方根距离，反映分子的空间延展程度。
- **计算**：从输入的三维坐标（`XYZ`）构建无键分子，调用 RDKit 的 `CalcRadiusOfGyration` 直接计算。
- **意义**：数值越大表示分子越松散或体积越大。

### 2. 非球形度 (Asphericity, `asphericity`)
- **定义**：基于惯量张量的形状描述符，量化分子偏离完美球体的程度。
- **计算**：使用 `CalcAsphericity`，返回值范围 [0, 1]，0 为完美球体，1 为完全非球形（如棒状）。
- **意义**：反映分子的各向异性。

### 3. 离心率 (Eccentricity, `eccentricity`)
- **定义**：描述分子椭球形状的拉伸程度，基于惯量张量的特征值。
- **计算**：使用 `CalcEccentricity`，0 对应球形，接近 1 对应极细长形状。
- **意义**：与 asphericity 类似，但更侧重于拉伸维度。

### 4. 手性中心数 (Chiral Centers, `num_chiral_centers`)
- **定义**：分子中四面体立体中心（手性碳等）的数量。
- **计算**：对 SMILES 加氢后，调用 `CalcNumAtomStereoCenters` 统计所有原子立体中心（包括 N、P 等）。
- **意义**：反映分子的立体化学复杂性，可能影响光学活性。

### 5. 顺反异构数 (Cis/Trans Isomers, `num_cis_trans`)
- **定义**：分子中具有 E/Z 构型的双键数量。
- **计算**：解析 SMILES，利用 `GetStereoInfo` 获取所有立体信息，计数类型为 `STEREO_E` 的条目（RDKit 中 `E` 表示双键立体）。
- **意义**：指示分子中刚性双键的立体多样性。

### 6. 波长 (Wavelength)
- **定义**：光谱测量时的激发或检测波长（单位 nm）。
- **计算**：直接从输入 JSON 的 `wavelength` 字段读取，可能为单值或列表，展开后每行对应一个波长。
- **意义**：作为光谱测量的条件参数，与 TPACS 值对应。

### 7. rg/分子量比值 (rg_mw_ratio)
- **定义**：回旋半径与分子量的比值，用于归一化大小与质量的关系。
- **计算**：`rg / MolWt`，其中 `MolWt` 由 SMILES 计算（未加氢）。
- **意义**：该比值可表示分子的“比体积”或“疏松度”，在药物化学中常与分子柔性相关。

### 8. 相对分子量 (Molecular Weight, `molecular_weight`)
- **定义**：分子的相对分子质量（g/mol）。
- **计算**：使用 RDKit 的 `Descriptors.MolWt` 基于 SMILES 直接计算（不添加氢）。
- **意义**：基本的分子属性，常作为尺度参数。

### 9. 刚性键比例 (Rigid Bond Ratio, `rigid_bond_ratio`)
- **定义**：分子中刚性化学键（包括双键、三键、环内单键和阻转单键（轴手性键））占所有非氢键的比例。
- **计算流程**：
  1. 对输入分子（由 SMILES 生成）进行加氢并生成构象（`transform_coordinate`）。
  2. 获取所有非氢键（原子序数 ≠ 1）作为分母。
  3. 识别刚性键：
     - **双键和三键**：直接检查键类型。
     - **环内单键**：利用 `ChiralAxialType5.get_single_bonds()` 返回的环内单键列表。
     - **阻转单键（轴手性键）**：调用 `get_chi_mat()` 获取轴手性键的原子对。
  4. 将以上各类键的索引（原子对）加入集合去重。
  5. 计算 `len(rigid_keys) / len(all_keys)`。
- **意义**：高比例表示分子整体刚性较强，可能影响构象多样性和分子与靶点的结合方式。

---

## 数据输出格式

所有特征与对应的 TPACS 值一起写入 `output.csv`，每一行代表一个分子在特定波长下的数据。特征列顺序如下：
