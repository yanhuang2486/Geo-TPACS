import json
import csv
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from quadrupole_utils import transform_coordinate
from quadrupole_v5 import ChiralAxialType5
from tqdm import tqdm  # 新增进度条


def get_stereo_info(mol):
    """兼容不同RDKit版本的GetStereoInfo调用"""
    if hasattr(Chem.rdmolops, 'GetStereoInfo'):
        return Chem.rdmolops.GetStereoInfo(mol)
    elif hasattr(Chem.rdchem, 'GetStereoInfo'):
        return Chem.rdchem.GetStereoInfo(mol)
    elif hasattr(mol, 'GetStereoInfo'):
        return mol.GetStereoInfo()
    else:
        print("Warning: GetStereoInfo not found, returning empty list")
        return []


def build_mol_from_coords(elements, xyz):
    """
    从元素符号列表和坐标列表构建一个只含原子和构象的RDKit分子（无键）。
    """
    if len(elements) != len(xyz):
        raise ValueError("元素数和坐标数不一致")
    rwmol = Chem.RWMol()
    for elem in elements:
        atom = Chem.Atom(elem.strip())
        rwmol.AddAtom(atom)
    mol = rwmol.GetMol()
    conf = Chem.Conformer(len(elements))
    for i, (x, y, z) in enumerate(xyz):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def count_chiral_centers(smiles):
    """从SMILES计算手性中心数量（使用CalcNumAtomStereoCenters，兼容性好）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    mol = Chem.AddHs(mol)
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
    return rdMolDescriptors.CalcNumAtomStereoCenters(mol)


def count_cis_trans(smiles):
    """从SMILES计算双键立体异构（顺反）的数量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
    stereo_infos = get_stereo_info(mol)
    count = 0
    for info in stereo_infos:
        if info.type == Chem.rdchem.StereoType.STEREO_E:
            count += 1
    return count


def compute_rigid_bond_ratio(mol):
    """
    计算分子中刚性键占所有化学键的比例（仅统计非氢键）。
    刚性键 = 双键 ∪ 三键 ∪ 环内单键 ∪ 阻转单键（轴手性键）。
    """
    # 1. 加氢并获取加氢分子
    mol_H, _ = transform_coordinate(mol)

    # 2. 初始化轴手性分析器
    axial = ChiralAxialType5(mol_H)

    # 3. 收集所有非氢键（即原始分子中的键）作为分母
    all_keys = set()
    for b in mol_H.GetBonds():
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        all_keys.add(tuple(sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])))

    rigid_keys = set()

    # 4. 双键和三键（仅非氢键）
    for b in mol_H.GetBonds():
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        bt = b.GetBondType()
        if bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            rigid_keys.add(tuple(sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])))

    # 5. 环内单键（来自 get_single_bonds，返回非环单键和环内单键）
    _, ring_bonds = axial.get_single_bonds()
    for b in ring_bonds:
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        rigid_keys.add(tuple(sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])))

    # 6. 阻转单键（轴手性键）
    result = axial.get_chi_mat()
    chiral_axes = result["chiral axes"]
    for a1, a2 in chiral_axes:
        atom1 = mol_H.GetAtomWithIdx(a1)
        atom2 = mol_H.GetAtomWithIdx(a2)
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue
        rigid_keys.add(tuple(sorted([a1, a2])))

    # 7. 计算比例（分母为非氢键总数）
    ratio = len(rigid_keys) / len(all_keys) if all_keys else 0.0
    return ratio


def main():
    input_file = "TPAML.json"
    output_file = "output.csv"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rows = []
    # 使用 tqdm 显示进度条
    for mol_data in tqdm(data, desc="Processing molecules"):
        compound_id = mol_data.get('compoundId', 'unknown')
        smiles = mol_data.get('molecule_smiles', '') or mol_data.get('smiles', '')
        elements = mol_data.get('element', [])
        xyz = mol_data.get('XYZ', [])

        # 兼容波长和TPACS可能为单值或列表
        wavelengths = mol_data.get('wavelength', [])
        tpacs = mol_data.get('TPACS', [])
        if not isinstance(wavelengths, list):
            wavelengths = [wavelengths]
        if not isinstance(tpacs, list):
            tpacs = [tpacs]

        # 数据完整性检查
        if not smiles or not elements or not xyz or not wavelengths or not tpacs:
            print(f"跳过 {compound_id}：数据不完整")
            continue
        if len(elements) != len(xyz):
            print(f"跳过 {compound_id}：元素数与坐标数不匹配")
            continue
        if len(wavelengths) != len(tpacs):
            print(f"跳过 {compound_id}：wavelength与TPACS长度不一致")
            continue

        # ---- 计算分子量（从未加氢的分子） ----
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"跳过 {compound_id}：无效的SMILES")
            continue
        mw = Descriptors.MolWt(mol)   # 浮点数

        # ---- 计算形状描述符（从坐标） ----
        try:
            mol_coords = build_mol_from_coords(elements, xyz)
            rg = rdMolDescriptors.CalcRadiusOfGyration(mol_coords)
            asp = rdMolDescriptors.CalcAsphericity(mol_coords)
            ecc = rdMolDescriptors.CalcEccentricity(mol_coords)
        except Exception as e:
            print(f"形状描述符计算失败 {compound_id}: {e}")
            rg = asp = ecc = None

        # ---- 计算 rg/mw 比值 ----
        if rg is not None and mw is not None and mw != 0:
            rg_mw_ratio = rg / mw
        else:
            rg_mw_ratio = None

        # ---- 计算立体化学信息（从SMILES） ----
        num_chiral = count_chiral_centers(smiles)
        num_cis_trans = count_cis_trans(smiles)

        # ---- 计算刚性键比例（利用SMILES和自动生成构象） ----
        try:
            rigid_ratio = compute_rigid_bond_ratio(mol)
        except Exception as e:
            print(f"刚性键比例计算失败 {compound_id}: {e}")
            rigid_ratio = None

        # 固定属性（新增 molecular_weight 和 rigid_bond_ratio）
        fixed = {
            'compoundId': compound_id,
            'rg': rg,
            'asphericity': asp,
            'eccentricity': ecc,
            'num_chiral_centers': num_chiral,
            'num_cis_trans': num_cis_trans,
            'rg_mw_ratio': rg_mw_ratio,
            'molecular_weight': mw,          # 新增列
            'rigid_bond_ratio': rigid_ratio
        }

        # 展开波长和TPACS
        for w, tp in zip(wavelengths, tpacs):
            row = fixed.copy()
            row['wavelength'] = w
            row['TPACS'] = tp
            rows.append(row)

    # 写入CSV（更新字段名）
    fieldnames = ['compoundId', 'rg', 'asphericity', 'eccentricity',
                  'num_chiral_centers', 'num_cis_trans', 'rg_mw_ratio',
                  'molecular_weight', 'rigid_bond_ratio', 'wavelength', 'TPACS']
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"完成，共写入 {len(rows)} 行数据到 {output_file}")


if __name__ == "__main__":
    main()