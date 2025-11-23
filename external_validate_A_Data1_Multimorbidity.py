# external_validate_A_Data1.py
import os
import json
from datetime import datetime
import copy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix
)
from joblib import load

warnings.filterwarnings("ignore")

# -------- Pretty table deps (optional) --------
try:
    from rich.console import Console
    from rich.table import Table

    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

try:
    from tabulate import tabulate

    _HAS_TABULATE = True
except Exception:
    _HAS_TABULATE = False


# ===========================
# 小工具：Specificity / NPV
# ===========================
def specificity_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan, np.nan
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return spec, npv


# ===========================
# Pretty table printer
# ===========================
def pretty_print_table(df, title=None, float_cols=None, float_digits=4):
    if float_cols is None:
        float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_show = df.copy()
    for c in float_cols:
        df_show[c] = df_show[c].astype(float).round(float_digits)

    if _HAS_RICH:
        console = Console()
        if title:
            console.rule(f"[bold]{title}")
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        for c in df_show.columns:
            align = "right" if c in float_cols else "left"
            table.add_column(str(c), justify=align, no_wrap=True)
        for _, row in df_show.iterrows():
            row_vals = []
            for c in df_show.columns:
                v = row[c]
                if pd.isna(v):
                    row_vals.append("-")
                elif c in float_cols:
                    row_vals.append(f"{float(v):.{float_digits}f}")
                else:
                    row_vals.append(str(v))
            table.add_row(*row_vals)
        console.print(table)
        return

    if _HAS_TABULATE:
        print(
            tabulate(
                df_show,
                headers="keys",
                tablefmt="github",
                showindex=False,
                floatfmt=f".{float_digits}f",
            )
        )
        return

    # fallback: 純文字對齊
    col_widths = {}
    for c in df_show.columns:
        max_val_len = max(
            [
                len(f"{v:.{float_digits}f}")
                if (c in float_cols and pd.notna(v))
                else len(str(v))
                for v in df_show[c]
            ]
            + [len(str(c))]
        )
        col_widths[c] = max_val_len
    if title:
        print("\n" + title)
    header = "  ".join(str(c).ljust(col_widths[c]) for c in df_show.columns)
    print(header)
    print("-" * len(header))
    for _, row in df_show.iterrows():
        parts = []
        for c in df_show.columns:
            v = row[c]
            if pd.isna(v):
                s = "-"
            elif c in float_cols:
                s = f"{float(v):.{float_digits}f}"
            else:
                s = str(v)
            align = str.ljust if c not in float_cols else str.rjust
            parts.append(align(s, col_widths[c]))
        print("  ".join(parts))


# ===========================
# 外部驗證專用 DataProcessor
# （邏輯要跟 A 組 training 一致）
# ===========================
class ExternalDataProcessorBaseline:
    """
    - 跟 A 組一樣：HRV: SDNN, LF, HF, LFHF + Age, Sex, BMI
    - 工程特徵：HRV_Mean, LF_HF_Ratio
    - 用「訓練時存下來的 outlier_bounds / scaler / imputer」做轉換
    """

    def __init__(self, treat_zero_as_missing_in_hrv=True):
        self.hrv_features = ["SDNN", "LF", "HF", "LFHF"]
        self.basic_features = ["Age", "Sex", "BMI"]
        self.log_hrv_cols = ["LF", "HF", "LFHF"]
        self.log_engineered_cols = ["HRV_Mean", "LF_HF_Ratio"]
        self.treat_zero_as_missing_in_hrv = treat_zero_as_missing_in_hrv

    def build_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """從 Data1 的原始欄位建立 HRV + basic + 工程特徵"""
        all_features = self.basic_features + self.hrv_features
        available = [f for f in all_features if f in df.columns]
        X = df[available].copy()

        hrv_cols = [c for c in self.hrv_features if c in X.columns]
        if len(hrv_cols) >= 3:
            X["HRV_Mean"] = X[hrv_cols].mean(axis=1)
        if "LF" in X.columns and "HF" in X.columns:
            X["LF_HF_Ratio"] = X["LF"] / (X["HF"] + 1e-6)
            X["LF_HF_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

        return X

    def _numeric_feature_list_for_outlier(self, X_frame: pd.DataFrame):
        candidates = []
        for col in (self.hrv_features + ["Age", "BMI"]):
            if col in X_frame.columns:
                candidates.append(col)
        for col in ["HRV_Mean", "LF_HF_Ratio"]:
            if col in X_frame.columns:
                candidates.append(col)
        out = []
        for c in candidates:
            s = pd.to_numeric(X_frame[c], errors="coerce")
            if s.notnull().any():
                out.append(c)
        return out

    def _apply_outlier_to_nan(self, X_frame: pd.DataFrame, outlier_bounds: dict):
        if not outlier_bounds:
            return X_frame

        Xp = X_frame.copy()

        # zero → NaN for HRV
        if self.treat_zero_as_missing_in_hrv:
            for col in [c for c in self.hrv_features if c in Xp.columns]:
                s = pd.to_numeric(Xp[col], errors="coerce")
                zero_mask = (s == 0)
                Xp.loc[zero_mask, col] = np.nan

        for col, (lb, ub) in outlier_bounds.items():
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors="coerce")
            mask = (s < lb) | (s > ub)
            Xp.loc[mask, col] = np.nan

        return Xp

    def _apply_log1p(self, X_frame: pd.DataFrame):
        Xp = X_frame.copy()
        for col in self.log_hrv_cols + self.log_engineered_cols:
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors="coerce")
            neg_mask = s < 0
            if neg_mask.any():
                Xp.loc[neg_mask, col] = np.nan
            Xp[col] = np.log1p(Xp[col])
        return Xp

    def apply_full_transform(
        self,
        X_raw: pd.DataFrame,
        feature_columns: list,
        outlier_bounds: dict,
        imputer: KNNImputer,
        scaler: StandardScaler,
    ) -> pd.DataFrame:
        """
        使用訓練時的 outlier_bounds / imputer / scaler
        對 Data1 做完整前處理，輸出欄位順序 = feature_columns
        """
        Xp = X_raw.copy()

        # 1) 補上缺的欄位，確保跟訓練時 feature_columns 一致
        for col in feature_columns:
            if col not in Xp.columns:
                Xp[col] = np.nan
        Xp = Xp[feature_columns].copy()

        # 2) 離群值 → NaN
        Xp = self._apply_outlier_to_nan(Xp, outlier_bounds)

        # 3) log1p
        Xp = self._apply_log1p(Xp)

        # 4) KNNImputer
        knn_f = self._numeric_feature_list_for_outlier(Xp)
        if imputer is not None and len(knn_f) > 0:
            try:
                Xp[knn_f] = imputer.transform(Xp[knn_f])
            except Exception as e:
                print(f"⚠️ KNNImputer transform 失敗，改用中位數補值: {e}")

        # 5) 仍有 NaN 就用 Data1 的中位數補
        if Xp.isnull().any().any():
            Xp.fillna(Xp.median(numeric_only=True), inplace=True)

        # 6) StandardScaler（跟 training 一樣，不處理 Sex）
        cols = Xp.columns.tolist()
        num_cols = [c for c in cols if c != "Sex" and pd.api.types.is_numeric_dtype(Xp[c])]
        other_cols = [c for c in cols if c not in num_cols]

        if scaler is not None and len(num_cols) > 0:
            X_num = pd.DataFrame(
                scaler.transform(Xp[num_cols]),
                columns=num_cols,
                index=Xp.index,
            )
        else:
            X_num = Xp[num_cols].copy()

        X_scaled = pd.concat([X_num, Xp[other_cols]], axis=1)[cols]
        return X_scaled


# ===========================
# 外部驗證主流程
# ===========================
def main():
    print("\n" + "=" * 70)
    print("🏁 外部驗證：使用 A 組最佳模型 → Data1 (External Test)")
    print("=" * 70)

    TRAIN_RUN_DIR = r"D:\FLY114-main\HRV-Project\data2_baseline"  # ← 你 A 組訓練好的資料夾
    FILE_PATH = r"D:\FLY114-main\data.xlsx"
    SHEET_NAME_TEST = "Data1"   # Data1 的工作表名稱
    # =====================================

    models_dir = os.path.join(TRAIN_RUN_DIR, "models")
    if not os.path.isdir(models_dir):
        print(f"❌ 找不到 models 目錄：{models_dir}")
        return

    # 建一個新的輸出資料夾存外部驗證結果
    timestamp = datetime.now().strftime("External_Data1_%Y%m%d_%H%M%S")
    out_dir = os.path.join(TRAIN_RUN_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"📂 外部驗證輸出資料夾: {out_dir}")

    # 讀 Data1
    try:
        df_test = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME_TEST)
        print(f"✓ Data1 載入成功: {df_test.shape[0]} 筆（工作表：{SHEET_NAME_TEST}）")
    except Exception as e:
        print(f"❌ 無法讀取 Data1：{e}")
        return

    label_names = ["Health", "SSD", "MDD", "Panic", "GAD"]
    processor = ExternalDataProcessorBaseline(treat_zero_as_missing_in_hrv=True)

    metrics_rows = []
    pred_sheets = {}

    for label in label_names:
        print("\n" + "-" * 60)
        print(f"🔍 處理標籤：{label}")

        meta_path = os.path.join(models_dir, f"{label}_best.json")
        if not os.path.isfile(meta_path):
            print(f"⚠️ 找不到 meta 檔：{meta_path}，略過此 label")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        feature_columns = meta.get("feature_columns", [])
        outlier_bounds = meta.get("outlier_bounds", {})
        threshold = float(meta.get("threshold", 0.5))
        files = meta.get("files", {})

        model_file = files.get("model")
        scaler_file = files.get("scaler")
        imputer_file = files.get("imputer")

        if not model_file:
            print(f"⚠️ {label}: meta 中沒有 model 檔名，略過")
            continue

        model_path = os.path.join(models_dir, model_file)
        if not os.path.isfile(model_path):
            print(f"⚠️ {label}: 找不到模型檔：{model_path}，略過")
            continue

        model = load(model_path)
        scaler = load(os.path.join(models_dir, scaler_file)) if scaler_file else None
        imputer = load(os.path.join(models_dir, imputer_file)) if imputer_file else None

        # ===== 準備 X_test_raw（Data1） =====
        X_raw = processor.build_raw_features(df_test)
        X_test = processor.apply_full_transform(
            X_raw=X_raw,
            feature_columns=feature_columns,
            outlier_bounds=outlier_bounds,
            imputer=imputer,
            scaler=scaler,
        )

        # 取 Data1 的真實標籤（如果有）
        if label in df_test.columns:
            y_true = df_test[label].astype(int).values
        else:
            y_true = None

        # ===== 推論 =====
        try:
            proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            # 有些模型（例如某些 Tree）可能沒有 predict_proba
            try:
                decision = model.decision_function(X_test)
                proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-8)
            except Exception as e:
                print(f"❌ {label}: 模型無法輸出機率：{e}")
                continue

        y_pred = (proba >= threshold).astype(int)

        # ===== 評估指標 =====
        if y_true is not None and len(np.unique(y_true)) > 1:
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, proba)
            except Exception:
                auc = np.nan
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            spec, npv = specificity_npv(y_true, y_pred)
        else:
            # 沒有真實標籤就只存預測結果
            f1 = acc = auc = precision = recall = spec = npv = np.nan

        metrics_rows.append(
            {
                "Label": label,
                "Threshold_used": threshold,
                "F1": f1,
                "Precision": precision,
                "Recall": recall,
                "Spec": spec,
                "NPV": npv,
                "AUC": auc,
                "ACC": acc,
            }
        )

        # 存每個病人的預測
        pred_df = pd.DataFrame(
            {
                "Index": df_test.index,
                f"{label}_y_true": y_true if y_true is not None else np.nan,
                f"{label}_y_pred": y_pred,
                f"{label}_proba": proba,
            }
        )
        pred_sheets[label] = pred_df

        print(
            f"   → {label:<6} | "
            f"F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, "
            f"Spec={spec:.4f}, NPV={npv:.4f}, AUC={auc:.4f}, ACC={acc:.4f}, t={threshold:.2f}"
        )

    # ====== 指標彙整輸出 ======
    if metrics_rows:
        metrics_df = pd.DataFrame(
            metrics_rows,
            columns=[
                "Label",
                "Threshold_used",
                "F1",
                "Precision",
                "Recall",
                "Spec",
                "NPV",
                "AUC",
                "ACC",
            ],
        )
        metrics_path = os.path.join(out_dir, "External_Data1_Metrics.xlsx")
        metrics_df.to_excel(metrics_path, index=False)
        pretty_print_table(
            metrics_df,
            title="External Validation on Data1 (A 組最佳模型)",
            float_cols=["F1", "Precision", "Recall", "Spec", "NPV", "AUC", "ACC"],
            float_digits=4,
        )
        print(f"\n✅ 外部驗證指標已輸出：{metrics_path}")
    else:
        print("⚠️ 沒有任何 label 成功計算指標。")

    # ====== 各 label 的個案預測輸出 ======
    if pred_sheets:
        preds_path = os.path.join(out_dir, "External_Data1_Predictions.xlsx")
        with pd.ExcelWriter(preds_path) as writer:
            for label, df_pred in pred_sheets.items():
                df_pred.to_excel(writer, sheet_name=label, index=False)
        print(f"✅ 個案層級預測已輸出：{preds_path}")

    print("\n🎉 外部驗證完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()