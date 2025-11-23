import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.inspection import permutation_importance  # Permutation importance
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold  # 目前未使用，但保留
from joblib import dump

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

# -------- SHAP --------
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


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
        sheet_name="data2",
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
        X_train_p = X_train.copy()
        X_test_p = X_test.copy() if X_test is not None else None

        if fit:
            self._fit_outlier_bounds(X_train_p)
        X_train_p = self._apply_outlier_to_nan(X_train_p, stage_note="Train")
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
# 分類器（擴充指標）
# ===========================
class ClassifierV61:
    def __init__(self, label_name, pos_count, neg_count, current_f1, target_f1):
        self.label_name = label_name
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.ratio = neg_count / pos_count if pos_count > 0 else 1
        self.current_f1 = current_f1
        self.target_f1 = target_f1
        self.gap = target_f1 - current_f1

        self.models = {}
        self.results = {}

        if self.gap > 0.10:
            self.strategy = "aggressive"
        elif self.gap > 0.05:
            self.strategy = "moderate"
        else:
            self.strategy = "conservative"

    def get_sampling_strategy(self):
        # 這裡維持你原本對各 label 的設定
        if self.label_name == "MDD":
            return "SMOTE", 0.75, 5
        if self.label_name == "Panic":
            return "BorderlineSMOTE", 0.55, 4
        if self.label_name == "GAD":
            return "SMOTE", 0.45, 5

        if self.pos_count < 100:
            sampler_type = "ADASYN"
            if self.strategy == "aggressive":
                sampling_ratio = 0.65
                k = 4
            else:
                sampling_ratio = 0.55
                k = 5
        else:
            if self.strategy == "aggressive":
                sampler_type = "SMOTE"
                sampling_ratio = 0.65
                k = 4
            elif self.strategy == "moderate":
                sampler_type = "SMOTE"
                sampling_ratio = 0.55
                k = 5
            else:
                sampler_type = "SMOTE"
                sampling_ratio = 0.50
                k = 5
        return sampler_type, sampling_ratio, k

    def build_models(self):
        scale_weight = int(self.ratio * 1.0)

        print("\n" + "=" * 70)
        print(f"🎯 {self.label_name}: F1={self.current_f1:.4f} → {self.target_f1:.4f}")
        print(f"   策略: {self.strategy.upper()}, 正例={self.pos_count}")

        if self.strategy == "aggressive":
            n_est = 700
            depth = 25
            lr = 0.03
            base_weight_mult = 2.0
        elif self.strategy == "moderate":
            n_est = 500
            depth = 18
            lr = 0.05
            base_weight_mult = 1.5
        else:
            n_est = 400
            depth = 12
            lr = 0.08
            base_weight_mult = 1.2

        if self.label_name == "MDD":
            weight_mult = 2.0
        elif self.label_name == "Panic":
            weight_mult = 1.8
        elif self.label_name == "GAD":
            weight_mult = 1.2
        else:
            weight_mult = base_weight_mult

        final_weight = max(1, int(scale_weight * weight_mult))

        if self.label_name == "Panic":
            self.models["XGB"] = xgb.XGBClassifier(
                n_estimators=650,
                max_depth=18,
                learning_rate=0.035,
                scale_pos_weight=final_weight,
                subsample=0.75,
                colsample_bytree=0.75,
                gamma=0.2,
                min_child_weight=1,
                reg_alpha=0.08,
                reg_lambda=0.6,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            self.models["XGB"] = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=int(depth * 0.4),
                learning_rate=lr,
                scale_pos_weight=final_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,
                min_child_weight=2,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )

        self.models["LGBM"] = lgb.LGBMClassifier(
            n_estimators=n_est,
            max_depth=int(depth * 0.4),
            learning_rate=lr,
            num_leaves=int(depth * 1.5),
            class_weight={0: 1, 1: final_weight},
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.models["RF"] = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight},
            random_state=42,
            n_jobs=-1,
        )

        self.models["ET"] = ExtraTreesClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight},
            random_state=42,
            n_jobs=-1,
        )

        self.models["GB"] = GradientBoostingClassifier(
            n_estimators=int(n_est * 0.6),
            max_depth=int(depth * 0.3),
            learning_rate=lr,
            subsample=0.8,
            min_samples_split=8,
            random_state=42,
        )

        self.models["BalancedRF"] = BalancedRandomForestClassifier(
            n_estimators=int(n_est * 0.8),
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )

        print(f"   ✓ 建立 {len(self.models)} 個模型, 權重={final_weight}")

    def _fit_single_model(self, name, model, X_resampled, y_resampled):
        if self.label_name in ["Panic", "MDD"]:
            use_early_stop = isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier))
            early_stop_rounds = 20
            val_split_ratio = 0.20
        else:
            use_early_stop = False
            early_stop_rounds = 30
            val_split_ratio = 0.15

        if use_early_stop:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=val_split_ratio, random_state=42
            )
            train_sub_idx, val_sub_idx = next(sss.split(X_resampled, y_resampled))
            X_tr_sub = X_resampled.iloc[train_sub_idx]
            y_tr_sub = y_resampled.iloc[train_sub_idx]
            X_val_sub = X_resampled.iloc[val_sub_idx]
            y_val_sub = y_resampled.iloc[val_sub_idx]
            try:
                if isinstance(model, xgb.XGBClassifier):
                    model.fit(
                        X_tr_sub,
                        y_tr_sub,
                        eval_set=[(X_val_sub, y_val_sub)],
                        early_stopping_rounds=early_stop_rounds,
                        verbose=False,
                    )
                elif isinstance(model, lgb.LGBMClassifier):
                    model.fit(
                        X_tr_sub,
                        y_tr_sub,
                        eval_set=[(X_val_sub, y_val_sub)],
                        eval_metric="binary_logloss",
                        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)],
                    )
            except TypeError:
                model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_resampled, y_resampled)
        return model

    def _optimize_threshold_precision_first(self, y_true, y_pred_proba, n_thresh=100):
        thresholds = np.linspace(0.10, 0.90, n_thresh)
        if self.label_name == "MDD":
            min_precision = 0.45
            min_recall = 0.45
        elif self.label_name == "Panic":
            min_precision = 0.45
            min_recall = 0.45
        elif self.label_name == "GAD":
            min_precision = 0.60
            min_recall = 0.30
        else:
            min_precision = 0.50
            min_recall = 0.30

        best_f1 = 0
        best_thresh = 0.5
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh

        if best_f1 == 0:
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                if y_pred.sum() == 0:
                    continue
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        return best_thresh, best_f1

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        sampler_type, sampling_ratio, k = self.get_sampling_strategy()
        print(f"\n   📈 採樣: {sampler_type} (ratio={sampling_ratio:.2f})...")
        try:
            if sampler_type == "ADASYN":
                sampler = ADASYN(
                    sampling_strategy=sampling_ratio, n_neighbors=k, random_state=42
                )
            elif sampler_type == "BorderlineSMOTE":
                sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42
                )
            else:
                sampler = SMOTE(
                    sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42
                )
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print(f"      正例: {y_train.sum()} → {y_resampled.sum()}")
        except Exception as e:
            print(f"      ⚠️ 採樣失敗: {e}")
            X_resampled, y_resampled = X_train, y_train

        print("\n   🔄 訓練與推論...")
        for name, model in self.models.items():
            try:
                fitted_model = self._fit_single_model(
                    name, model, X_resampled, y_resampled
                )
                y_pred_proba = fitted_model.predict_proba(X_test)[:, 1]
                best_thresh, _ = self._optimize_threshold_precision_first(
                    y_test, y_pred_proba
                )
                y_pred = (y_pred_proba >= best_thresh).astype(int)

                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except Exception:
                    auc = np.nan
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                spec, npv = specificity_npv(y_test, y_pred)

                self.results[name] = {
                    "f1_score": f1,
                    "accuracy": acc,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall,
                    "specificity": spec,
                    "npv": npv,
                    "threshold": best_thresh,
                    "y_pred": y_pred,
                    "y_pred_proba": y_pred_proba,
                    "y_true": y_test.values,
                    "model": fitted_model,
                }

                status = "✅" if f1 >= self.target_f1 else "⚠️"
                print(
                    f"      {name:13s}: F1={f1:.4f} {status}, "
                    f"P={precision:.3f}, R={recall:.3f}, "
                    f"Spec={spec:.3f}, NPV={npv:.3f}, "
                    f"AUC={auc:.3f}, ACC={acc:.3f}, t={best_thresh:.2f}"
                )

            except Exception as e:
                print(f"      ❌ {name}: {e}")

        self._create_ensemble(X_test, y_test)
        return self.results

    def _create_ensemble(self, X_test, y_test):
        if len(self.results) < 2:
            return
        try:
            predictions, weights = [], []
            for name, r in self.results.items():
                if self.label_name == "Panic":
                    weight = 0.5 * r["f1_score"] + 0.5 * r["precision"]
                elif self.label_name == "MDD":
                    weight = 0.4 * r["f1_score"] + 0.6 * r["recall"]
                else:
                    weight = r["f1_score"] * (0.5 + 0.5 * r["precision"])
                predictions.append(r["y_pred_proba"])
                weights.append(max(weight, 0.01))

            weights = np.array(weights)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)
            ensemble_proba = np.average(predictions, axis=0, weights=weights)
            best_thresh, _ = self._optimize_threshold_precision_first(
                y_test, ensemble_proba
            )
            ensemble_pred = (ensemble_proba >= best_thresh).astype(int)

            f1 = f1_score(y_test, ensemble_pred)
            acc = accuracy_score(y_test, ensemble_pred)
            try:
                auc = roc_auc_score(y_test, ensemble_proba)
            except Exception:
                auc = np.nan
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            spec, npv = specificity_npv(y_test, ensemble_pred)

            self.results["Ensemble"] = {
                "f1_score": f1,
                "accuracy": acc,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "specificity": spec,
                "npv": npv,
                "threshold": best_thresh,
                "y_pred": ensemble_pred,
                "y_pred_proba": ensemble_proba,
                "y_true": y_test.values,
            }

            status = "✅" if f1 >= self.target_f1 else "⚠️"
            print(
                f"      {'Ensemble':13s}: F1={f1:.4f} {status}, "
                f"P={precision:.3f}, R={recall:.3f}, "
                f"Spec={spec:.3f}, NPV={npv:.3f}, "
                f"AUC={auc:.3f}, ACC={acc:.3f}"
            )

        except Exception as e:
            print(f"      ⚠️ 集成失敗: {e}")

    def get_best_model(self):
        if not self.results:
            return None, None
        best_name = max(
            self.results.items(), key=lambda x: (x[1]["f1_score"], x[1]["precision"])
        )[0]
        return best_name, self.results[best_name]


# ===========================
# 只保存「每個 label 整體最好的單一模型」
# ===========================
def save_best_model(
    models_dir,
    label,
    model_obj,
    scaler,
    imputer,
    feature_columns,
    outlier_bounds,
    threshold,
):
    os.makedirs(models_dir, exist_ok=True)
    base = f"{label}_best"
    model_path = os.path.join(models_dir, base + ".joblib")
    scaler_path = os.path.join(models_dir, base + "_scaler.joblib")
    imputer_path = os.path.join(models_dir, base + "_imputer.joblib")
    meta_path = os.path.join(models_dir, base + ".json")

    dump(model_obj, model_path)
    if scaler is not None:
        dump(scaler, scaler_path)
    if imputer is not None:
        dump(imputer, imputer_path)

    meta = {
        "label": label,
        "threshold": float(threshold),
        "feature_columns": list(feature_columns),
        "outlier_bounds": outlier_bounds,
        "files": {
            "model": os.path.basename(model_path),
            "scaler": os.path.basename(scaler_path) if scaler is not None else None,
            "imputer": os.path.basename(imputer_path) if imputer is not None else None,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存整體最佳模型：{model_path}")


# ===========================
# 主流程：SSD/MDD/Panic/GAD vs Health
# ===========================
def main():
    print("\n" + "=" * 70)
    print("🏁 HRV+Demo：SSD/MDD/Panic/GAD 病人 vs 真健康人（只保留每個 label 的整體最佳模型）")
    print("=" * 70)

    timestamp = datetime.now().strftime("Run_Patient_vs_Health_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"📂 輸出資料夾: {run_dir}")

    FILE_PATH = r"D:\FLY114-main\data.xlsx"
    SHEET_NAME = "Data2"  # 或 "Filled_AllData"

    processor = DataProcessorBaseline(
        FILE_PATH, SHEET_NAME, iqr_multiplier=3.0, treat_zero_as_missing_in_hrv=True
    )

    if not processor.load_data():
        return
    if not processor.prepare_features_and_labels():
        return

    X_full = processor.X
    df_full = processor.df  # 包含 Health, SSD, MDD, Panic, GAD 原始標籤

    # 只做四個「病人 vs 健康」任務
    label_names = ["SSD", "MDD", "Panic", "GAD"]

    # 統計用容器
    all_fold_results = {lb: [] for lb in label_names}
    oof_true = {lb: [] for lb in label_names}
    oof_proba = {lb: [] for lb in label_names}
    oof_pred = {lb: [] for lb in label_names}

    # 每個 label 的整體最佳單一模型
    overall_best = {
        lb: {
            "model_name": None,
            "f1": -1.0,
            "precision": 0.0,
            "recall": 0.0,
            "spec": 0.0,
            "npv": 0.0,
            "auc": 0.0,
            "acc": 0.0,
            "threshold": 0.5,
            "model_obj": None,
            "scaler": None,
            "imputer": None,
            "outlier_bounds": None,
            "feature_columns": None,
            "fold": None,
        }
        for lb in label_names
    }

    # RF 模型資訊，供 Avg folds importance / SHAP 用
    rf_fold_info = {lb: [] for lb in label_names}

    # baseline / target F1 設定（仍然可以沿用原來估計值）
    baseline_f1_map = {
        "Health": 0.7940,
        "SSD": 0.6667,
        "MDD": 0.4691,
        "Panic": 0.5021,
        "GAD": 0.5797,
    }
    target_f1_map = {
        "Health": 0.80,
        "SSD": 0.65,
        "MDD": 0.50,
        "Panic": 0.55,
        "GAD": 0.65,
    }

    # ===========================
    # 每個 label：病人 vs 真健康人
    # ===========================
    for label in label_names:
        print("\n" + "#" * 70)
        print(f"🔍 任務：{label} 病人 vs 真健康人 (Health)")
        print("#" * 70)

        if "Health" not in df_full.columns or label not in df_full.columns:
            print(f"❌ 缺少 Health 或 {label} 欄位，略過")
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

        health_mask = df_sub["Health"] == 1
        label_mask = df_sub[label] == 1

        y_sub = np.where(label_mask, 1, 0)  # 1=病人(label)，0=健康(Health)

        n_pos = int((y_sub == 1).sum())
        n_neg = int((y_sub == 0).sum())
        print(f"   樣本數：總共 {len(y_sub)} 筆（{label}=1 病人 {n_pos}，Health=1 健康 {n_neg}）")

        if n_pos < 10 or n_neg < 10:
            print(f"   ⚠️ {label}: 病人或健康人數過少，結果可能不穩定")

        # F1 目標設定（沿用原 baseline / target）
        current_baseline = baseline_f1_map.get(label, 0.5)
        target_goal = target_f1_map.get(label, 0.7)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_id = 1
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            print(f"\n📂 {label} - Fold {fold_id}/5")
            print("-" * 70)

            X_train_raw = X_sub.iloc[train_idx]
            X_test_raw = X_sub.iloc[test_idx]
            y_train = pd.Series(y_sub[train_idx], index=X_train_raw.index)
            y_test = pd.Series(y_sub[test_idx], index=X_test_raw.index)

            # 預處理（每個 fold 重新 fit）
            X_train_p, X_test_p = processor.impute_and_scale(
                X_train_raw, X_test_raw, fit=True
            )
            print(f"✓ 預處理完成（{label} - Fold {fold_id}）")

            pos = int(y_train.sum())
            neg = len(y_train) - pos

            trainer = ClassifierV61(label, pos, neg, current_baseline, target_goal)
            trainer.build_models()
            results = trainer.train_and_evaluate(X_train_p, X_test_p, y_train, y_test)

            # 顯示用（Ensemble 優先）
            if "Ensemble" in results:
                show_res = results["Ensemble"]
                show_from = "Ensemble"
            else:
                base_models = [m for m in results.keys() if m != "Ensemble"]
                best_name_fold = max(
                    base_models, key=lambda m: results[m]["f1_score"]
                )
                show_res = results[best_name_fold]
                show_from = best_name_fold

            all_fold_results[label].append(
                {
                    "f1_score": show_res["f1_score"],
                    "precision": show_res["precision"],
                    "recall": show_res["recall"],
                    "specificity": show_res["specificity"],
                    "npv": show_res["npv"],
                    "auc": show_res["auc"],
                    "accuracy": show_res["accuracy"],
                }
            )

            # OOF 累積（這裡選用 Ensemble 或該 fold 的最佳模型）
            chosen_name = "Ensemble" if "Ensemble" in results else best_name_fold
            chosen_res = results[chosen_name]
            oof_true[label].extend(list(chosen_res["y_true"]))
            oof_proba[label].extend(list(chosen_res["y_pred_proba"]))
            oof_pred[label].extend(list(chosen_res["y_pred"]))

            # 更新整體最佳單一模型（不含 Ensemble）
            base_models_only = [m for m in results.keys() if m != "Ensemble"]
            for mname in base_models_only:
                r = results[mname]
                if r["f1_score"] > overall_best[label]["f1"]:
                    overall_best[label] = {
                        "model_name": mname,
                        "f1": float(r["f1_score"]),
                        "precision": float(r["precision"]),
                        "recall": float(r["recall"]),
                        "spec": float(r["specificity"]),
                        "npv": float(r["npv"]),
                        "auc": float(r["auc"]),
                        "acc": float(r["accuracy"]),
                        "threshold": float(r["threshold"]),
                        "model_obj": r["model"],
                        "scaler": copy.deepcopy(processor.scaler),
                        "imputer": copy.deepcopy(processor.knn_imputer),
                        "outlier_bounds": copy.deepcopy(processor.outlier_bounds_),
                        "feature_columns": list(processor.X.columns),
                        "fold": fold_id,
                    }

            # 記錄 RF 模型資訊（給 Avg folds importance / SHAP）
            if "RF" in results:
                r_rf = results["RF"]
                rf_fold_info[label].append(
                    {
                        "model": r_rf["model"],
                        "X_test": X_test_p.copy(),
                        "y_test": y_test.copy(),
                    }
                )

            print(
                f"   → {label:<8} | "
                f"F1={show_res['f1_score']:.4f}, "
                f"P={show_res['precision']:.4f}, "
                f"R={show_res['recall']:.4f}, "
                f"Spec={show_res['specificity']:.4f}, "
                f"NPV={show_res['npv']:.4f}, "
                f"AUC={show_res['auc']:.4f}, "
                f"ACC={show_res['accuracy']:.4f}  [{show_from}]"
            )

            fold_id += 1

    # =============== 最終彙整、只保存每個 label 的整體最佳單一模型 ===============
    print("\n" + "=" * 70)
    print("🏁 5-Fold 最終結果（SSD/MDD/Panic/GAD 病人 vs 真健康人）")
    print("=" * 70)

    rows = []
    plot_labels, plot_best_f1, plot_avg_f1 = [], [], []
    plot_avg_precision, plot_avg_recall = [], []
    plot_stability_mean, plot_stability_std = [], []
    roc_curves_oof = {}

    for label in label_names:
        ob = overall_best[label]
        if ob["model_obj"] is not None:
            save_best_model(
                models_dir=models_dir,
                label=label,
                model_obj=ob["model_obj"],
                scaler=ob["scaler"],
                imputer=ob["imputer"],
                feature_columns=ob["feature_columns"],
                outlier_bounds=ob["outlier_bounds"],
                threshold=ob["threshold"],
            )
            print(
                f"⭐ {label}: 整體最佳單一模型 = {ob['model_name']}（來自 fold {ob['fold']}），F1={ob['f1']:.4f}"
            )
        else:
            print(f"⚠️ {label}: 沒有成功訓練到任何模型")

        if len(all_fold_results[label]) == 0:
            continue

        temp_df = pd.DataFrame(all_fold_results[label])
        f1_avg = float(temp_df["f1_score"].mean())
        p_avg = float(temp_df["precision"].mean())
        r_avg = float(temp_df["recall"].mean())
        spec_avg = float(temp_df["specificity"].mean())
        npv_avg = float(temp_df["npv"].mean())
        auc_avg = float(temp_df["auc"].mean())
        acc_avg = float(temp_df["accuracy"].mean())

        rows.append(
            {
                "Label": label,
                "BestModel": ob["model_name"],
                "Fold(Best)": ob["fold"],
                "F1(Best)": ob["f1"],
                "P(Best)": ob["precision"],
                "R(Best)": ob["recall"],
                "Spec(Best)": ob["spec"],
                "NPV(Best)": ob["npv"],
                "AUC(Best)": ob["auc"],
                "ACC(Best)": ob["acc"],
                "F1(avg)": f1_avg,
                "P(avg)": p_avg,
                "R(avg)": r_avg,
                "Spec(avg)": spec_avg,
                "NPV(avg)": npv_avg,
                "AUC(avg)": auc_avg,
                "ACC(avg)": acc_avg,
            }
        )

        plot_labels.append(label)
        plot_best_f1.append(ob["f1"])
        plot_avg_f1.append(f1_avg)
        plot_avg_precision.append(p_avg)
        plot_avg_recall.append(r_avg)

        fold_f1_list = [fold_res["f1_score"] for fold_res in all_fold_results[label]]
        stability_mean = float(np.mean(fold_f1_list)) if len(fold_f1_list) > 0 else np.nan
        stability_std = (
            float(np.std(fold_f1_list, ddof=1)) if len(fold_f1_list) > 1 else 0.0
        )
        plot_stability_mean.append(stability_mean)
        plot_stability_std.append(stability_std)

        # ROC OOF
        y_true_all = np.array(oof_true[label])
        y_proba_all = np.array(oof_proba[label])
        if len(np.unique(y_true_all)) > 1 and y_proba_all.size > 0:
            try:
                fpr_oof, tpr_oof, _ = roc_curve(y_true_all, y_proba_all)
                auc_oof = roc_auc_score(y_true_all, y_proba_all)
            except Exception:
                fpr_oof, tpr_oof, auc_oof = None, None, np.nan
        else:
            fpr_oof, tpr_oof, auc_oof = None, None, np.nan
        roc_curves_oof[label] = (fpr_oof, tpr_oof, auc_oof)

    # =============== 輸出 Excel（含 Spec/NPV） ===============
    if rows:
        out_df = pd.DataFrame(
            rows,
            columns=[
                "Label",
                "BestModel",
                "Fold(Best)",
                "F1(Best)",
                "P(Best)",
                "R(Best)",
                "Spec(Best)",
                "NPV(Best)",
                "AUC(Best)",
                "ACC(Best)",
                "F1(avg)",
                "P(avg)",
                "R(avg)",
                "Spec(avg)",
                "NPV(avg)",
                "AUC(avg)",
                "ACC(avg)",
            ],
        )
        excel_path = os.path.join(run_dir, "Patient_vs_Health_KFold_Best_and_Avg.xlsx")
        out_df.to_excel(excel_path, index=False)
        print(f"\n✅ 已輸出結果至 {excel_path}")

        cols_show = [
            "Label",
            "BestModel",
            "F1(Best)",
            "P(Best)",
            "R(Best)",
            "Spec(Best)",
            "NPV(Best)",
            "AUC(Best)",
            "ACC(Best)",
            "F1(avg)",
            "P(avg)",
            "R(avg)",
            "Spec(avg)",
            "NPV(avg)",
            "AUC(avg)",
            "ACC(avg)",
        ]
        pretty_print_table(
            out_df[cols_show],
            title="5-Fold 最終結果（病人 vs 真健康人）",
            float_cols=[c for c in cols_show if c not in ["Label", "BestModel"]],
            float_digits=4,
        )
    else:
        print("⚠️ 沒有任何 label 產生結果，請檢查資料或條件")

    # ============================
    # 🔍 RF feature importance + SHAP + Permutation (Avg folds)
    # ============================
    print("\n" + "=" * 70)
    print("🔍 RandomForest Feature Importance + SHAP + Permutation（各標籤, Avg folds）")
    print("=" * 70)

    rf_importance_xlsx = os.path.join(run_dir, "Patient_vs_Health_RF_Importances_AvgFolds.xlsx")
    with pd.ExcelWriter(rf_importance_xlsx) as writer:
        for label in label_names:
            fold_list = rf_fold_info[label]
            if not fold_list:
                print(f"⚠️ {label}: 無 RF 模型可用，略過")
                continue

            print(f"[{label}] 計算 RF importance / Permutation / SHAP (Avg folds)...")

            feature_cols = list(fold_list[0]["X_test"].columns)
            n_features = len(feature_cols)

            rf_imp_list = []
            perm_mean_list = []
            perm_std_list = []
            shap_mean_list = []
            shap_all_values = []
            shap_all_X = []

            for fold_idx, info in enumerate(fold_list, 1):
                rf_model = info["model"]
                X_test_rf = info["X_test"][feature_cols]
                y_test_rf = info["y_test"]

                # 1) feature_importances_
                fi = getattr(rf_model, "feature_importances_", None)
                if fi is None:
                    fi_vals = np.zeros(n_features)
                else:
                    fi_vals = np.asarray(fi).reshape(-1)
                if len(fi_vals) != n_features:
                    fi_vals = fi_vals[:n_features]
                rf_imp_list.append(fi_vals)

                # 2) permutation importance
                try:
                    perm = permutation_importance(
                        rf_model,
                        X_test_rf,
                        y_test_rf,
                        n_repeats=20,
                        random_state=42,
                        n_jobs=-1,
                        scoring="f1",
                    )
                    pm = np.asarray(perm.importances_mean).reshape(-1)
                    ps = np.asarray(perm.importances_std).reshape(-1)
                except Exception as e:
                    print(f"   ⚠️ {label} fold{fold_idx}: permutation importance 失敗: {e}")
                    pm = np.zeros(n_features)
                    ps = np.zeros(n_features)
                if len(pm) != n_features:
                    pm = pm[:n_features]
                if len(ps) != n_features:
                    ps = ps[:n_features]
                perm_mean_list.append(pm)
                perm_std_list.append(ps)

                # 3) SHAP
                if _HAS_SHAP:
                    try:
                        X_for_shap = X_test_rf
                        if len(X_for_shap) > 400:
                            X_for_shap = X_for_shap.sample(400, random_state=42)

                        explainer = shap.TreeExplainer(rf_model)
                        shap_values = explainer.shap_values(X_for_shap)
                        if isinstance(shap_values, list):  # binary: 取正類
                            shap_vals_pos = shap_values[1]
                        else:
                            shap_vals_pos = shap_values
                        shap_mean = np.mean(np.abs(shap_vals_pos), axis=0)
                        shap_mean = np.asarray(shap_mean).reshape(-1)
                        if len(shap_mean) != n_features:
                            shap_mean = shap_mean[:n_features]
                        shap_mean_list.append(shap_mean)

                        shap_all_values.append(shap_vals_pos)
                        shap_all_X.append(X_for_shap[feature_cols])
                    except Exception as e:
                        print(f"   ⚠️ {label} fold{fold_idx}: SHAP 計算失敗: {e}")

            rf_imp_avg = (
                np.mean(rf_imp_list, axis=0) if rf_imp_list else np.zeros(n_features)
            )
            perm_mean_avg = (
                np.mean(perm_mean_list, axis=0)
                if perm_mean_list
                else np.zeros(n_features)
            )
            perm_std_avg = (
                np.mean(perm_std_list, axis=0)
                if perm_std_list
                else np.zeros(n_features)
            )
            shap_mean_avg = (
                np.mean(shap_mean_list, axis=0)
                if shap_mean_list
                else np.zeros(n_features)
            )

            imp_df = pd.DataFrame(
                {
                    "Feature": feature_cols,
                    "RF_importance_mean": rf_imp_avg,
                    "Permutation_mean": perm_mean_avg,
                    "Permutation_std": perm_std_avg,
                    "SHAP_mean_abs": shap_mean_avg,
                }
            ).sort_values("Permutation_mean", ascending=False)

            # 寫進 Excel
            imp_df.to_excel(writer, sheet_name=label, index=False)

            # 終端顯示前 10 名
            pretty_print_table(
                imp_df.head(10),
                title=f"{label} - RF / Permutation / SHAP (Top 10, Avg folds)",
                float_cols=[
                    "RF_importance_mean",
                    "Permutation_mean",
                    "Permutation_std",
                    "SHAP_mean_abs",
                ],
                float_digits=4,
            )

            # 圖 1：Permutation Importance (Avg folds)
            top_k = min(25, len(imp_df))
            top_df = imp_df.head(top_k).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                top_df["Feature"],
                top_df["Permutation_mean"],
                xerr=top_df["Permutation_std"],
            )
            ax.set_xlabel("Permutation Importance (mean ± std)")
            ax.set_title(f"Permutation Importance (Avg folds) - {label}")
            ax.axvline(0, color="black", linewidth=0.8)
            fig.tight_layout()
            perm_path = os.path.join(
                run_dir, f"PermutationImportance_AvgFolds_{label}.png"
            )
            fig.savefig(perm_path, dpi=300)
            plt.close(fig)
            print(f"📊 已輸出 Permutation 圖: {perm_path}")

            # 圖 2：SHAP Summary (Avg folds)
            if _HAS_SHAP and shap_all_values:
                try:
                    shap_vals_all = np.concatenate(shap_all_values, axis=0)
                    X_all = pd.concat(shap_all_X, axis=0)[feature_cols]

                    fig = plt.figure(figsize=(7, 10))
                    shap.summary_plot(
                        shap_vals_all,
                        X_all,
                        show=False,
                        max_display=min(25, n_features),
                    )
                    plt.title(f"SHAP Summary (Avg folds) - {label}")
                    fig.tight_layout()
                    shap_path = os.path.join(
                        run_dir, f"SHAP_Summary_AvgFolds_{label}.png"
                    )
                    fig.savefig(shap_path, dpi=300)
                    plt.close(fig)
                    print(f"📊 已輸出 SHAP Summary 圖: {shap_path}")
                except Exception as e:
                    print(f"   ⚠️ {label}: SHAP summary 繪圖失敗: {e}")
            else:
                if not _HAS_SHAP:
                    print(f"   ⚠️ {label}: 未安裝 shap，略過 SHAP Summary 圖")
                else:
                    print(f"   ⚠️ {label}: 無 SHAP 資料，略過 SHAP Summary 圖")

    print(f"\n✅ RF importance / SHAP / Permutation (Avg folds) 已輸出至: {rf_importance_xlsx}")

    # ============================
    # 其他視覺化：F1 / PR / Stability / ROC
    # ============================
    def attach_value_labels(bars, ax):
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

    if plot_labels:
        x_idx = np.arange(len(plot_labels))
        bar_w = 0.35

        # (1) F1-score 柱狀圖
        fig, ax = plt.subplots(figsize=(10, 6))
        bars_avg = ax.bar(
            x_idx - bar_w / 2,
            plot_avg_f1,
            width=bar_w,
            label="F1 avg (5-fold / chosen per fold)",
        )
        bars_best = ax.bar(
            x_idx + bar_w / 2,
            plot_best_f1,
            width=bar_w,
            label="F1 Best Single Model (overall)",
        )
        ax.set_xticks(x_idx)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel("F1-score")
        ax.set_title("F1-score by Label (Patient vs Health)")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        attach_value_labels(bars_avg, ax)
        attach_value_labels(bars_best, ax)
        fig.tight_layout()
        f1_bar_path = os.path.join(run_dir, "F1_BarChart_Patient_vs_Health.png")
        fig.savefig(f1_bar_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出: {f1_bar_path}")

        # (2) Precision vs Recall 散點圖
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl, p_val, r_val in zip(plot_labels, plot_avg_precision, plot_avg_recall):
            ax.scatter(r_val, p_val, s=80)
            ax.text(r_val + 0.01, p_val + 0.01, lbl)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Recall (per label, Patient vs Health)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        pr_scatter_path = os.path.join(run_dir, "Precision_Recall_Scatter_Patient_vs_Health.png")
        fig.savefig(pr_scatter_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出: {pr_scatter_path}")

        # (3) 5-Fold 穩定度圖（誤差棒）
        fig, ax = plt.subplots(figsize=(10, 6))
        bars_stab = ax.bar(x_idx, plot_stability_mean, yerr=plot_stability_std, capsize=5)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(plot_labels)
        ax.set_ylabel("F1-score (mean ± std)")
        ax.set_title("5-Fold Stability of F1-score (Patient vs Health)")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        attach_value_labels(bars_stab, ax)
        fig.tight_layout()
        stability_path = os.path.join(run_dir, "F1_Stability_ErrorBar_Patient_vs_Health.png")
        fig.savefig(stability_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出: {stability_path}")

        # (4) ROC Curve（OOF）
        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl in plot_labels:
            fpr_oof, tpr_oof, auc_oof = roc_curves_oof.get(lbl, (None, None, np.nan))
            if fpr_oof is not None and tpr_oof is not None:
                ax.plot(fpr_oof, tpr_oof, label=f"{lbl} (AUC={auc_oof:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (OOF, Patient vs Health)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(run_dir, "ROC_Curves_OOF_Patient_vs_Health.png")
        fig.savefig(roc_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出: {roc_path}")

    print(
        "\n✅ 完成！SSD/MDD/Panic/GAD 病人 vs 真健康人 的 HRV+Demo 分類，"
        "並輸出各 label 的 RF importance / SHAP / permutation (Avg folds)，檔案位置:",
        run_dir,
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
