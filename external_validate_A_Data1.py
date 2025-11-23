import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')

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

# -------- joblib (load models) --------
from joblib import load as joblib_load


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
# 資料處理（Baseline：HRV+Demo-Only）
# ===========================
class DataProcessorBaseline:
    """
    Baseline：
    - 僅使用兩資料集皆有的 HRV 與人口學特徵
      HRV: SDNN, LF, HF, LFHF
      Demo: Age, Sex, BMI
    - 工程特徵：HRV_Mean、LF_HF_Ratio（若 LF/HF 皆存在）
    """

    def __init__(
        self,
        file_path,
        sheet_name="Data2",
        iqr_multiplier: float = 3.0,
        treat_zero_as_missing_in_hrv: bool = True,
    ):
        self.file_path = file_path
        self.sheet_name = sheet_name

        self.hrv_features = ["SDNN", "LF", "HF", "LFHF"]
        self.basic_features = ["Age", "Sex", "BMI"]
        self.label_names = ["Health", "SSD", "MDD", "Panic", "GAD"]

        self.log_hrv_cols = ["LF", "HF", "LFHF"]
        self.log_engineered_cols = ["HRV_Mean", "LF_HF_Ratio"]

        self.iqr_multiplier = iqr_multiplier
        self.treat_zero_as_missing_in_hrv = treat_zero_as_missing_in_hrv

        self.knn_imputer = None
        self.scaler = None
        self.outlier_bounds_ = None
        self.df = None
        self.X = None
        self.y_dict = {}

    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            print(f"✓ 載入: {self.df.shape[0]} 筆（工作表：{self.sheet_name}）")
            return True
        except Exception as e:
            print(f"❌ {e}")
            return False

    def prepare_features_and_labels(self):
        all_features = self.basic_features + self.hrv_features
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()

        print("\n🔨 Baseline 特徵工程（HRV+Demo-Only）...")

        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3:
            self.X["HRV_Mean"] = self.X[hrv_cols].mean(axis=1)
        if "LF" in self.X.columns and "HF" in self.X.columns:
            self.X["LF_HF_Ratio"] = self.X["LF"] / (self.X["HF"] + 1e-6)
            self.X["LF_HF_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

        for label in self.label_names:
            if label in self.df.columns:
                self.y_dict[label] = self.df[label].copy()

        print(f"✓ 特徵數量: {self.X.shape[1]}（含工程特徵）")
        return len(self.y_dict) > 0

    def _numeric_feature_list_for_outlier(self, X_frame: pd.DataFrame):
        candidates = []
        for col in (self.hrv_features + ["Age", "BMI"]):
            if col in X_frame.columns:
                candidates.append(col)
        for col in ["HRV_Mean", "LF_HF_Ratio"]:
        # add engineered features if present
            if col in X_frame.columns:
                candidates.append(col)
        out = []
        for c in candidates:
            s = pd.to_numeric(X_frame[c], errors="coerce")
            if s.notnull().any():
                out.append(c)
        return out

    def _compute_iqr_bounds(self, s: pd.Series, k: float):
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lower = s.quantile(0.001)
            upper = s.quantile(0.999)
        else:
            lower = q1 - k * iqr
            upper = q3 + k * iqr
        return float(lower), float(upper)

    def _fit_outlier_bounds(self, X_train: pd.DataFrame):
        # 外部驗證不會用到 (不重新 fit)，但保留原本程式結構
        num_cols = self._numeric_feature_list_for_outlier(X_train)
        bounds = {}
        for col in num_cols:
            s = pd.to_numeric(X_train[col], errors="coerce")
            lower, upper = self._compute_iqr_bounds(s.dropna(), self.iqr_multiplier)
            bounds[col] = (lower, upper)
        self.outlier_bounds_ = bounds

    def _apply_outlier_to_nan(self, X_frame: pd.DataFrame, stage_note: str = ""):
        if not self.outlier_bounds_:
            return X_frame

        Xp = X_frame.copy()
        total_flagged = 0

        if self.treat_zero_as_missing_in_hrv:
            for col in [c for c in self.hrv_features if c in Xp.columns]:
                s = pd.to_numeric(Xp[col], errors="coerce")
                zero_mask = s == 0
                total_flagged += int(zero_mask.sum())
                Xp.loc[zero_mask, col] = np.nan

        for col, (lb, ub) in self.outlier_bounds_.items():
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors="coerce")
            mask = (s < lb) | (s > ub)
            total_flagged += int(mask.sum())
            Xp.loc[mask, col] = np.nan

        if stage_note:
            print(f"   • [{stage_note}] 離群值→NaN：{total_flagged} 個")
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

    def impute_and_scale(self, X_train, X_test=None, fit=True):
        """
        外部驗證時會用：
        - fit=False
        - 已經先把 self.outlier_bounds_, self.knn_imputer, self.scaler 填成訓練時保存的物件
        """
        X_train_p = X_train.copy()
        X_test_p = X_test.copy() if X_test is not None else None

        if fit:
            self._fit_outlier_bounds(X_train_p)
        X_train_p = self._apply_outlier_to_nan(X_train_p, stage_note="Train" if fit else "")
        if X_test_p is not None:
            X_test_p = self._apply_outlier_to_nan(X_test_p, stage_note="Test")

        X_train_p = self._apply_log1p(X_train_p)
        if X_test_p is not None:
            X_test_p = self._apply_log1p(X_test_p)

        knn_f = self._numeric_feature_list_for_outlier(X_train_p)
        if len(knn_f) > 0 and X_train_p[knn_f].isnull().any().any():
            if fit or (self.knn_imputer is None):
                self.knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
                X_train_p[knn_f] = self.knn_imputer.fit_transform(X_train_p[knn_f])
            else:
                X_train_p[knn_f] = self.knn_imputer.transform(X_train_p[knn_f])
            if X_test_p is not None:
                X_test_p[knn_f] = self.knn_imputer.transform(X_test_p[knn_f])

        if X_train_p.isnull().any().any():
            X_train_p.fillna(X_train_p.median(numeric_only=True), inplace=True)
            if X_test_p is not None:
                X_test_p.fillna(X_train_p.median(numeric_only=True), inplace=True)

        cols = X_train_p.columns.tolist()
        num_cols = [c for c in cols if c != "Sex" and pd.api.types.is_numeric_dtype(X_train_p[c])]
        other_cols = [c for c in cols if c not in num_cols]

        if fit or (self.scaler is None):
            self.scaler = StandardScaler()
            X_train_num = pd.DataFrame(
                self.scaler.fit_transform(X_train_p[num_cols]),
                columns=num_cols,
                index=X_train_p.index,
            )
        else:
            X_train_num = pd.DataFrame(
                self.scaler.transform(X_train_p[num_cols]),
                columns=num_cols,
                index=X_train_p.index,
            )

        X_train_s = pd.concat([X_train_num, X_train_p[other_cols]], axis=1)[cols]

        if X_test_p is not None:
            X_test_num = pd.DataFrame(
                self.scaler.transform(X_test_p[num_cols]),
                columns=num_cols,
                index=X_test_p.index,
            )
            X_test_s = pd.concat([X_test_num, X_test_p[other_cols]], axis=1)[cols]
            return X_train_s, X_test_s

        return X_train_s


# ===========================
# 載入訓練好的最佳模型與 Meta
# ===========================
def load_best_model_and_meta(models_dir, label):
    """
    從訓練 Run 的 models 資料夾載入:
    - model (xxx_best.joblib)
    - scaler (xxx_best_scaler.joblib)
    - imputer (xxx_best_imputer.joblib)
    - meta json (含 threshold, feature_columns, outlier_bounds)
    """
    base = f"{label}_best"
    model_path = os.path.join(models_dir, base + ".joblib")
    scaler_path = os.path.join(models_dir, base + "_scaler.joblib")
    imputer_path = os.path.join(models_dir, base + "_imputer.joblib")
    meta_path = os.path.join(models_dir, base + ".json")

    if not os.path.isfile(model_path) or not os.path.isfile(meta_path):
        print(f"❌ {label}: 找不到模型或 meta 檔案，略過。")
        return None

    model = joblib_load(model_path)

    scaler = None
    if os.path.isfile(scaler_path):
        scaler = joblib_load(scaler_path)

    imputer = None
    if os.path.isfile(imputer_path):
        imputer = joblib_load(imputer_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    threshold = meta.get("threshold", 0.5)
    feature_columns = meta.get("feature_columns", [])
    outlier_bounds = meta.get("outlier_bounds", {})

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "outlier_bounds": outlier_bounds,
    }


# ===========================
# 主流程：External Validation on Data1
# ===========================
def main():
    print("\n" + "=" * 70)
    print("🏁 HRV+Demo External Validation：SSD/MDD/Panic/GAD 病人 vs 真健康人 (Data1)")
    print("=" * 70)

    # 訓練 Run 的根目錄（內含 models/ 與 data.xlsx）
    BASE_RUN_DIR = r"D:\FLY114-MAIN\Run_Patient_vs_Health_20251121_230903"
    MODELS_DIR = os.path.join(BASE_RUN_DIR, "models")

    # 外部驗證用資料檔案與工作表
    EXCEL_NAME = "D:\FLY114-MAIN\data.xlsx"  # 若實際檔名不同，請在這裡修改
    FILE_PATH_TEST = os.path.join(BASE_RUN_DIR, EXCEL_NAME)
    SHEET_NAME_TEST = "Data1"

    timestamp = datetime.now().strftime("External_Validation_Data1_%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_RUN_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"📂 外部驗證輸出資料夾: {out_dir}")

    # 準備 Data1 baseline 特徵
    processor = DataProcessorBaseline(
        FILE_PATH_TEST,
        sheet_name=SHEET_NAME_TEST,
        iqr_multiplier=3.0,
        treat_zero_as_missing_in_hrv=True,
    )

    if not processor.load_data():
        print("❌ Data1 載入失敗，結束。")
        return
    if not processor.prepare_features_and_labels():
        print("❌ Data1 缺少必要標籤欄位，結束。")
        return

    X_full = processor.X
    df_full = processor.df  # 包含 Health, SSD, MDD, Panic, GAD 等原始標籤

    label_names = ["SSD", "MDD", "Panic", "GAD"]

    # 統計容器
    rows = []
    plot_labels = []
    plot_f1 = []
    roc_curves = {}

    for label in label_names:
        print("\n" + "#" * 70)
        print(f"🔍 外部驗證任務：{label} 病人 vs 真健康人 (Health) - Data1")
        print("#" * 70)

        # 載入對應 label 的最佳模型 + preprocessor meta
        info = load_best_model_and_meta(MODELS_DIR, label)
        if info is None:
            continue

        model = info["model"]
        scaler = info["scaler"]
        imputer = info["imputer"]
        threshold = float(info["threshold"])
        feature_columns = info["feature_columns"]
        outlier_bounds = info["outlier_bounds"]

        # 檢查 Data1 是否有 Health / label 欄位
        if "Health" not in df_full.columns or label not in df_full.columns:
            print(f"❌ Data1 缺少 Health 或 {label} 欄位，略過。")
            continue

        # 定義「有效」列：Health=1 或 該 label=1，且兩者都是 0/1，有效
        mask_valid = (
            df_full["Health"].isin([0, 1]) &
            df_full[label].isin([0, 1]) &
            ((df_full["Health"] == 1) | (df_full[label] == 1))
        )
        df_sub = df_full.loc[mask_valid].copy()
        X_sub = X_full.loc[mask_valid].copy()

        # 保險：排除 Health=1 且 label=1 的奇怪情況（XOR）
        mask_xor = (df_sub["Health"] == 1) ^ (df_sub[label] == 1)
        df_sub = df_sub.loc[mask_xor]
        X_sub = X_sub.loc[mask_xor]

        if len(df_sub) == 0:
            print(f"⚠️ {label}: Data1 無可用樣本，略過。")
            continue

        health_mask = df_sub["Health"] == 1
        label_mask = df_sub[label] == 1
        y = np.where(label_mask, 1, 0)

        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"   Data1 樣本數：總共 {len(y)} 筆（{label}=1 病人 {n_pos}，Health=1 健康 {n_neg}）")

        if n_pos < 5 or n_neg < 5:
            print(f"   ⚠️ {label}: Data1 病人或健康樣本數偏少，指標可能不穩定。")

        # 讓 Data1 的特徵順序/欄位與訓練時一致（缺的補 NaN，多的自動被 drop）
        X_eval_raw = X_sub.copy()
        for col in feature_columns:
            if col not in X_eval_raw.columns:
                X_eval_raw[col] = np.nan
        X_eval_raw = X_eval_raw[feature_columns]

        # 使用訓練時保存的 preprocessor 物件
        processor.outlier_bounds_ = outlier_bounds
        processor.knn_imputer = imputer
        processor.scaler = scaler

        # impute_and_scale with fit=False: 不重新 fit，只套用訓練時的 outlier/imputer/scaler
        X_eval_processed = processor.impute_and_scale(X_eval_raw, X_test=None, fit=False)
        print(f"✓ Data1 前處理完成（{label}）")

        # 推論
        y_true = y
        try:
            y_proba = model.predict_proba(X_eval_processed)[:, 1]
        except Exception as e:
            print(f"❌ {label}: predict_proba 失敗：{e}")
            continue

        y_pred = (y_proba >= threshold).astype(int)

        # 指標計算
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        spec, npv = specificity_npv(y_true, y_pred)

        try:
            auc_val = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
        except Exception:
            auc_val = np.nan

        print(
            f"   → Data1 External ({label}) | "
            f"F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, "
            f"Spec={spec:.4f}, NPV={npv:.4f}, AUC={auc_val:.4f}, ACC={acc:.4f}, t={threshold:.2f}"
        )

        # ROC curve
        if len(np.unique(y_true)) > 1:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
            except Exception:
                fpr, tpr = None, None
        else:
            fpr, tpr = None, None

        roc_curves[label] = (fpr, tpr, auc_val)

        # Summary row
        rows.append(
            {
                "Label": label,
                "n_total": int(len(y_true)),
                "n_patient": n_pos,
                "n_health": n_neg,
                "F1": float(f1),
                "Precision": float(precision),
                "Recall": float(recall),
                "Specificity": float(spec),
                "NPV": float(npv),
                "AUC": float(auc_val),
                "Accuracy": float(acc),
                "Threshold_used": float(threshold),
            }
        )

        plot_labels.append(label)
        plot_f1.append(f1)

    # ======== 輸出 Excel 結果 ========
    if rows:
        result_df = pd.DataFrame(
            rows,
            columns=[
                "Label",
                "n_total",
                "n_patient",
                "n_health",
                "F1",
                "Precision",
                "Recall",
                "Specificity",
                "NPV",
                "AUC",
                "Accuracy",
                "Threshold_used",
            ],
        )
        excel_path = os.path.join(out_dir, "External_Validation_Data1_Patient_vs_Health.xlsx")
        result_df.to_excel(excel_path, index=False)
        print(f"\n✅ 已輸出外部驗證結果至 {excel_path}")

        pretty_print_table(
            result_df,
            title="External Validation on Data1（SSD/MDD/Panic/GAD vs Health）",
            float_cols=[
                "F1", "Precision", "Recall", "Specificity",
                "NPV", "AUC", "Accuracy", "Threshold_used"
            ],
            float_digits=4,
        )
    else:
        print("⚠️ 沒有任何 label 完成外部驗證，請檢查模型或 Data1 資料。")
        return

    # ======== 圖形：F1 柱狀圖 ========
    if plot_labels:
        x = np.arange(len(plot_labels))
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(x, plot_f1)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1-score")
        ax.set_title("External Validation F1-score by Label (Data1, Patient vs Health)")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        fig.tight_layout()
        f1_path = os.path.join(plots_dir, "External_F1_BarChart_Data1_Patient_vs_Health.png")
        fig.savefig(f1_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出外部驗證 F1 圖: {f1_path}")

    # ======== 圖形：ROC Curves ========
    fig, ax = plt.subplots(figsize=(8, 6))
    has_any_curve = False
    for label in label_names:
        fpr, tpr, auc_val = roc_curves.get(label, (None, None, np.nan))
        if fpr is not None and tpr is not None:
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})")
            has_any_curve = True

    if has_any_curve:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("External Validation ROC Curves (Data1, Patient vs Health)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(plots_dir, "External_ROC_Curves_Data1_Patient_vs_Health.png")
        fig.savefig(roc_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出外部驗證 ROC 圖: {roc_path}")
    else:
        plt.close(fig)
        print("⚠️ 無法繪製 ROC 曲線（可能某些 label 在 Data1 只有單一類別）。")

    print(
        "\n✅ 外部驗證完成！Data1 的 SSD/MDD/Panic/GAD 病人 vs 真健康人 "
        f"結果已輸出至: {out_dir}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()