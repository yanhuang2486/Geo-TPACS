from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm

# ============================================================
# 0. 文件配置：把这些文件放在同一文件夹，必要时修改文件名
# ============================================================
ROOT = Path(__file__).resolve().parent

JSON_FILE = ROOT / "TPAML.json"
ROSTER_FILE = ROOT / "TPA_856_0307.csv"          # 作者给出的856个SMILES+TPACS，顺序即模型顺序
FEATURE94_FILE = ROOT / "cleaned_TPAML_Features_94.csv"
FEATURE696_FILE = ROOT / "cleaned_TPAML_Features_696.csv"
OUTPUT_DIR = ROOT / "tpaml_unified_output"

# 原论文最终6个特征
BASE6_FEATURES = [
    "Conju-Max-Distance",
    "Wavelength (Exp nm)",
    "ET(30) (Solvent)",
    "PEOE-Charge-Max",
    "Atomic-LogP-Min",
    "Atomic-MR-Max",
]

SPATIAL_FEATURES = [
    "rg",
    "asphericity",
    "eccentricity",
    "num_chiral_centers",
    "num_cis_trans",
    "rg_mw_ratio",
    "molecular_weight",
    "rigid_bond_ratio",
]

# 第三位同学的轴手性依赖是可选的。
# 文件齐全时使用其原算法；缺少时退化为“双/三键+环内键”的RDKit版本，并在输出中记录方法。
try:
    from quadrupole_utils import transform_coordinate
    from quadrupole_v5 import ChiralAxialType5

    HAS_AXIAL_DEPENDENCIES = True
except ImportError:
    HAS_AXIAL_DEPENDENCIES = False


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def canonical_smiles(smiles: str) -> str:
    """
    返回用于跨文件匹配的稳定SMILES键。

    大多数分子使用RDKit canonical SMILES；少数作者数据中的超大/特殊
    SMILES无法被当前RDKit kekulize，此时退回到去除空白后的原始SMILES。
    由于roster和TPAML.json来自同一数据源，原始字符串仍可稳定一一匹配。
    """
    text = "".join(str(smiles).split())
    if not text:
        raise ValueError("SMILES为空")

    # sanitize=False先避免RDKit在读取阶段直接抛出kekulize错误；随后尝试规范化。
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return "CAN:" + Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    # 对无法解析的特殊SMILES使用原始字符串键，不再中断整条流水线。
    warnings.warn(f"RDKit无法规范化一个SMILES，将使用原始字符串匹配：{text[:80]}...")
    return "RAW:" + text



def parse_smiles_robust(smiles: str) -> Chem.Mol:
    """
    尽可能稳健地解析SMILES。
    常规解析失败时，保留芳香键而跳过kekulize，以兼容少数超大共轭体系。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("无效SMILES")

    mol.UpdatePropertyCache(strict=False)
    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops, catchErrors=True)
    Chem.GetSymmSSSR(mol)
    return mol

def build_mol_from_coords(elements: list[Any], xyz: list[Any]) -> Chem.Mol:
    """根据JSON中的元素和XYZ构建带构象的无键RDKit分子，用于3D形状描述符。"""
    if len(elements) != len(xyz):
        raise ValueError("元素数与XYZ坐标数不一致")

    rw_mol = Chem.RWMol()
    for elem in elements:
        rw_mol.AddAtom(Chem.Atom(str(elem).strip()))

    mol = rw_mol.GetMol()
    conf = Chem.Conformer(len(elements))
    for i, coord in enumerate(xyz):
        x, y, z = map(float, coord[:3])
        conf.SetAtomPosition(i, (x, y, z))
    mol.AddConformer(conf)
    return mol


def count_chiral_centers(smiles: str) -> int:
    try:
        mol = parse_smiles_robust(smiles)
    except Exception:
        return 0
    return int(rdMolDescriptors.CalcNumAtomStereoCenters(mol))


def count_cis_trans(smiles: str) -> int:
    """统计SMILES中已经明确指定为E或Z的双键。"""
    try:
        mol = parse_smiles_robust(smiles)
    except Exception:
        return 0
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            count += 1
    return count


def rigid_bond_ratio_rdkit(mol: Chem.Mol) -> float:
    """无自定义轴手性模块时的稳健退化版本：双键、三键、环内键/全部非氢键。"""
    heavy_bonds = []
    rigid_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        heavy_bonds.append(bond)
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE) or bond.IsInRing():
            rigid_bonds.append(bond)
    return len(rigid_bonds) / len(heavy_bonds) if heavy_bonds else 0.0


def rigid_bond_ratio_colleague(mol: Chem.Mol) -> float:
    """第三位同学原始定义：双/三键、环内单键、轴手性阻转单键。"""
    mol_h, _ = transform_coordinate(mol)
    axial = ChiralAxialType5(mol_h)

    all_keys: set[tuple[int, int]] = set()
    for bond in mol_h.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        all_keys.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))

    rigid_keys: set[tuple[int, int]] = set()
    for bond in mol_h.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            rigid_keys.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))

    _, ring_bonds = axial.get_single_bonds()
    for bond in ring_bonds:
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        rigid_keys.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))

    result = axial.get_chi_mat()
    for a1, a2 in result.get("chiral axes", []):
        atom1 = mol_h.GetAtomWithIdx(a1)
        atom2 = mol_h.GetAtomWithIdx(a2)
        if atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1:
            continue
        rigid_keys.add(tuple(sorted((a1, a2))))

    return len(rigid_keys) / len(all_keys) if all_keys else 0.0


def compute_spatial_features(record: dict[str, Any]) -> dict[str, Any]:
    smiles = record["smiles"]
    mol = parse_smiles_robust(smiles)

    elements = record.get("element", [])
    xyz = record.get("XYZ", [])
    mol_coords = build_mol_from_coords(elements, xyz)

    rg = float(rdMolDescriptors.CalcRadiusOfGyration(mol_coords))
    asphericity = float(rdMolDescriptors.CalcAsphericity(mol_coords))
    eccentricity = float(rdMolDescriptors.CalcEccentricity(mol_coords))
    molecular_weight = float(Descriptors.MolWt(mol))

    if HAS_AXIAL_DEPENDENCIES:
        try:
            rigid_ratio = float(rigid_bond_ratio_colleague(mol))
            rigid_method = "colleague_axial_method"
        except Exception as exc:
            # 自定义轴手性算法对部分复杂分子会触发AssertionError或返回值版本不兼容。
            # 仅对该分子退化到稳定RDKit定义，保留全部856个样本，并记录回退原因。
            rigid_ratio = float(rigid_bond_ratio_rdkit(mol))
            rigid_method = f"rdkit_fallback_after_{type(exc).__name__}"
    else:
        rigid_ratio = float(rigid_bond_ratio_rdkit(mol))
        rigid_method = "rdkit_fallback_without_axial_bonds"

    return {
        "rg": rg,
        "asphericity": asphericity,
        "eccentricity": eccentricity,
        "num_chiral_centers": count_chiral_centers(smiles),
        "num_cis_trans": count_cis_trans(smiles),
        "rg_mw_ratio": rg / molecular_weight if molecular_weight else np.nan,
        "molecular_weight": molecular_weight,
        "rigid_bond_ratio": rigid_ratio,
        "rigid_bond_method": rigid_method,
    }


def load_roster_856() -> pd.DataFrame:
    """
    建立856个建模样本的权威顺序：
    - TPA_856_0307.csv 提供SMILES和所选TPACS；
    - 94维表提供同一行对应的实验波长和values_ln。
    """
    roster = pd.read_csv(ROSTER_FILE, header=None, names=["smiles", "TPACS"])
    df94 = pd.read_csv(FEATURE94_FILE)

    if len(roster) != 856 or len(df94) != 856:
        raise ValueError(f"样本数异常：roster={len(roster)}, feature94={len(df94)}，预期均为856")

    required = {"Wavelength (Exp nm)", "values_ln"}
    missing = required - set(df94.columns)
    if missing:
        raise KeyError(f"94维表缺少列：{sorted(missing)}")

    roster.insert(0, "model_row", np.arange(856, dtype=int))
    roster["wavelength"] = pd.to_numeric(df94["Wavelength (Exp nm)"], errors="raise")
    roster["values_ln"] = pd.to_numeric(df94["values_ln"], errors="raise")

    # 三重校验：values_ln必须等于ln(TPACS)
    if not np.allclose(np.exp(roster["values_ln"]), roster["TPACS"], rtol=1e-8, atol=1e-8):
        raise ValueError("94维表values_ln与TPA_856_0307.csv中的TPACS不能逐行对应")

    roster["canonical_smiles"] = roster["smiles"].map(canonical_smiles)
    return roster


def build_json_lookup(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for record in data:
        smiles = record.get("smiles") or record.get("molecule_smiles")
        if not smiles:
            continue
        try:
            key = canonical_smiles(smiles)
        except Exception:
            continue
        lookup.setdefault(key, []).append(record)
    return lookup


def match_roster_to_json(roster: pd.DataFrame, data: list[dict[str, Any]]) -> pd.DataFrame:
    """按 canonical SMILES + 波长 + TPACS，将856行逐一定位回TPAML.json。"""
    lookup = build_json_lookup(data)
    mapping_rows: list[dict[str, Any]] = []

    for row in tqdm(roster.itertuples(index=False), total=len(roster), desc="Matching 856 samples"):
        candidates = lookup.get(row.canonical_smiles, [])
        exact_matches: list[tuple[dict[str, Any], int]] = []

        for record in candidates:
            wavelengths = as_list(record.get("wavelength", []))
            tpacs_values = as_list(record.get("TPACS", []))
            if len(wavelengths) != len(tpacs_values):
                continue
            for point_index, (w, tp) in enumerate(zip(wavelengths, tpacs_values)):
                if math.isclose(float(w), float(row.wavelength), rel_tol=0, abs_tol=1e-8) and math.isclose(
                    float(tp), float(row.TPACS), rel_tol=0, abs_tol=1e-8
                ):
                    exact_matches.append((record, point_index))

        if len(exact_matches) != 1:
            raise ValueError(
                f"第{row.model_row}行无法唯一映射：匹配数={len(exact_matches)}, "
                f"SMILES={row.smiles}, wavelength={row.wavelength}, TPACS={row.TPACS}"
            )

        record, point_index = exact_matches[0]
        element_set = sorted({str(e).strip() for e in record.get("element", [])})
        mapping_rows.append(
            {
                "model_row": int(row.model_row),
                "Old_index_1": record.get("Old_index_1"),
                "smiles": record.get("smiles") or record.get("molecule_smiles"),
                "canonical_smiles": row.canonical_smiles,
                "doi": record.get("doi", ""),
                "compoundId": record.get("compoundId", ""),
                "Solvent": record.get("Solvent", ""),
                "selected_point_index": point_index,
                "wavelength": float(row.wavelength),
                "TPACS": float(row.TPACS),
                "values_ln": float(row.values_ln),
                "elements": ",".join(element_set),
            }
        )

    mapping = pd.DataFrame(mapping_rows).sort_values("model_row").reset_index(drop=True)

    if len(mapping) != 856 or mapping["model_row"].nunique() != 856:
        raise AssertionError("映射表未得到856个唯一model_row")
    if mapping["Old_index_1"].isna().any():
        raise AssertionError("映射表存在缺失Old_index_1")
    if mapping["Old_index_1"].duplicated().any():
        warnings.warn("映射中存在重复Old_index_1，请检查是否同一分子被作者重复收录。")

    return mapping


def compute_geometry_856(mapping: pd.DataFrame, data: list[dict[str, Any]]) -> pd.DataFrame:
    by_old_index = {record.get("Old_index_1"): record for record in data}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in tqdm(mapping.itertuples(index=False), total=len(mapping), desc="Computing spatial features"):
        record = by_old_index.get(row.Old_index_1)
        if record is None:
            raise KeyError(f"JSON中找不到Old_index_1={row.Old_index_1}")

        try:
            features = compute_spatial_features(record)
            rows.append(
                {
                    "model_row": row.model_row,
                    "Old_index_1": row.Old_index_1,
                    "smiles": row.smiles,
                    **features,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "model_row": row.model_row,
                    "Old_index_1": row.Old_index_1,
                    "smiles": row.smiles,
                    "error": repr(exc),
                }
            )

    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_DIR / "spatial_feature_failures.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            f"有{len(failures)}个分子空间特征计算失败。详见 spatial_feature_failures.csv，"
            "为保证全链路统一，程序不会静默删除这些样本。"
        )

    geometry = pd.DataFrame(rows).sort_values("model_row").reset_index(drop=True)
    if geometry.shape[0] != 856:
        raise AssertionError(f"空间特征表行数={len(geometry)}，预期856")
    return geometry


def merge_model_inputs(mapping: pd.DataFrame, geometry: pd.DataFrame) -> None:
    df94 = pd.read_csv(FEATURE94_FILE)
    df696 = pd.read_csv(FEATURE696_FILE)
    if len(df94) != 856 or len(df696) != 856:
        raise ValueError("94维或696维表不是856行")

    # 给纯特征矩阵显式补上model_row，再做一对一合并，杜绝隐式按行猜测。
    df94 = df94.copy()
    df696 = df696.copy()
    df94.insert(0, "model_row", np.arange(856, dtype=int))
    df696.insert(0, "model_row", np.arange(856, dtype=int))

    id_cols = [
        "model_row",
        "Old_index_1",
        "smiles",
        "canonical_smiles",
        "doi",
        "compoundId",
        "Solvent",
        "selected_point_index",
        "wavelength",
        "TPACS",
        "values_ln",
    ]
    identity = mapping[id_cols].copy()

    geom_model = geometry[["model_row", "Old_index_1"] + SPATIAL_FEATURES + ["rigid_bond_method"]].copy()
    identity_geom = identity.merge(
        geom_model,
        on=["model_row", "Old_index_1"],
        how="inner",
        validate="one_to_one",
    )

    # 校验目标和波长未发生错位
    if not np.allclose(identity_geom["values_ln"], df94["values_ln"]):
        raise AssertionError("mapping与94维表的values_ln错位")
    if not np.allclose(identity_geom["wavelength"], df94["Wavelength (Exp nm)"]):
        raise AssertionError("mapping与94维表的波长错位")

    # 输出1：全身份+空间特征，可追溯
    identity_geom.to_csv(OUTPUT_DIR / "sample_index_and_spatial_856.csv", index=False, encoding="utf-8-sig")

    # 输出2：6个论文基准特征 + 8个空间特征 + target，直接输入模型
    missing6 = [c for c in BASE6_FEATURES if c not in df94.columns]
    if missing6:
        raise KeyError(f"94维表缺少论文6特征：{missing6}")

    six_plus_spatial = identity_geom[
        ["model_row", "Old_index_1", "smiles", "canonical_smiles", "doi", "compoundId"]
        + SPATIAL_FEATURES
    ].merge(
        df94[["model_row"] + BASE6_FEATURES + ["values_ln"]],
        on="model_row",
        how="inner",
        validate="one_to_one",
    )
    ordered_cols = (
        ["model_row", "Old_index_1", "smiles", "canonical_smiles", "doi", "compoundId"]
        + BASE6_FEATURES
        + SPATIAL_FEATURES
        + ["values_ln"]
    )
    six_plus_spatial = six_plus_spatial[ordered_cols]
    six_plus_spatial.to_csv(OUTPUT_DIR / "model_input_6_plus_spatial_856.csv", index=False, encoding="utf-8-sig")

    # 输出3/4：94或696特征 + 空间特征，供扩展实验
    geom_only = geometry[["model_row"] + SPATIAL_FEATURES]
    model94 = df94.merge(geom_only, on="model_row", how="inner", validate="one_to_one")
    model696 = df696.merge(geom_only, on="model_row", how="inner", validate="one_to_one")
    model94.to_csv(OUTPUT_DIR / "model_input_94_plus_spatial_856.csv", index=False, encoding="utf-8-sig")
    model696.to_csv(OUTPUT_DIR / "model_input_696_plus_spatial_856.csv", index=False, encoding="utf-8-sig")

    # 输出审计报告
    audit = {
        "json_records": 929,
        "model_samples": 856,
        "mapping_unique_model_row": int(mapping["model_row"].nunique()),
        "mapping_unique_old_index": int(mapping["Old_index_1"].nunique()),
        "spatial_rows": int(len(geometry)),
        "rigid_bond_method": sorted(geometry["rigid_bond_method"].unique().tolist()),
        "six_plus_spatial_shape": list(six_plus_spatial.shape),
        "feature94_plus_spatial_shape": list(model94.shape),
        "feature696_plus_spatial_shape": list(model696.shape),
        "all_targets_aligned": bool(np.allclose(identity_geom["values_ln"], df94["values_ln"])),
        "all_wavelengths_aligned": bool(
            np.allclose(identity_geom["wavelength"], df94["Wavelength (Exp nm)"])
        ),
    }
    with open(OUTPUT_DIR / "pipeline_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)


def main() -> None:
    for path in (JSON_FILE, ROSTER_FILE, FEATURE94_FILE, FEATURE696_FILE):
        require_file(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(data) != 929:
        warnings.warn(f"TPAML.json含{len(data)}个对象，不是预期的929个")

    print("[1/4] 建立作者856个建模样本的权威顺序……")
    roster = load_roster_856()

    print("[2/4] 将856行逐一映射回TPAML.json……")
    mapping = match_roster_to_json(roster, data)
    mapping.to_csv(OUTPUT_DIR / "sample_index_856.csv", index=False, encoding="utf-8-sig")

    print("[3/4] 对同一批856个分子计算空间结构特征……")
    geometry = compute_geometry_856(mapping, data)
    geometry.to_csv(OUTPUT_DIR / "spatial_features_856.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 合并并生成可直接输入模型的数据……")
    merge_model_inputs(mapping, geometry)

    print("\n完成。输出目录：", OUTPUT_DIR)
    print("核心训练文件：model_input_6_plus_spatial_856.csv")
    print("索引审计文件：sample_index_856.csv")
    if not HAS_AXIAL_DEPENDENCIES:
        print("警告：未检测到quadrupole_utils.py和quadrupole_v5.py，刚性键比例使用RDKit退化版本。")


if __name__ == "__main__":
    main()
