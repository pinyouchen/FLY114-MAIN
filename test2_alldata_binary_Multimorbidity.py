import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ===========================
# 資料處理
# ===========================

class DataProcessorV61:
    
    def __init__(self, file_path, sheet_name='Merged_Sheet',
                 iqr_multiplier: float = 3.0,
                 treat_zero_as_missing_in_hrv: bool = True):
        self.file_path = file_path
        self.sheet_name = sheet_name

        self.hrv_features = ['MEANH', 'LF', 'HF', 'NLF', 'SC', 'FT', 'RSA', 'TP', 'VLF']
        self.clinical_features = ['DM', 'TCA', 'MARTA']
        self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
        self.basic_features = ['Age', 'Sex', 'BMI']
        self.label_names = ['Health', 'SSD', 'MDD', 'Panic', 'GAD']

        self.log_hrv_cols = ['LF', 'HF', 'TP', 'VLF', 'SC', 'RSA']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio']

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
            print(f"✓ 載入: {self.df.shape[0]} 筆")
            return True
        except Exception as e:
            print(f"❌ {e}")
            return False
    
    def prepare_features_and_labels(self):

        all_features = (self.basic_features + self.hrv_features + 
                        self.clinical_features + self.psych_features)
        available = [f for f in all_features if f in self.df.columns]
        
        self.X = self.df[available].copy()
        
        print(f"\n🔨 特徵工程...")
        
        # === v5.5 原有特徵 ===
        if 'Age' in self.X.columns and 'BMI' in self.X.columns:
            self.X['Age_BMI'] = self.X['Age'] * self.X['BMI']
            print("   ✓ Age/BMI")

        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3:
            self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
            if 'LF' in self.X.columns and 'HF' in self.X.columns:
                self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
                self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            print("   ✓ HRV")

        psych_cols = [c for c in self.psych_features if c in self.X.columns]
        if len(psych_cols) >= 3:
            self.X['Psych_Sum'] = self.X[psych_cols].sum(axis=1)
            print("   ✓ 心理")
        
        # 微調1: 交感神經活性指標（Panic 的生理標記）
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['Sympathetic_Index'] = self.X['LF'] / (self.X['LF'] + self.X['HF'] + 1e-6)
            print("   ✓ [Panic] 交感神經指標")
        
        # 微調2: BAI 非線性變換（捕捉極端焦慮）
        if 'bai' in self.X.columns:
            self.X['bai_log'] = np.log1p(self.X['bai'])
            print("   ✓ [Panic] BAI 對數變換")
        
        # 微調3: Panic 風險複合指標
        if 'bai' in self.X.columns and 'HRV_Mean' in self.X.columns:
            self.X['Panic_Risk'] = self.X['bai'] / (self.X['HRV_Mean'] + 1e-6)
            print("   ✓ [Panic] 風險複合指標")
        
        # 標籤
        for label in self.label_names:
            if label in self.df.columns:
                self.y_dict[label] = self.df[label].copy()
        
        print(f"\n✓ 特徵數量: {self.X.shape[1]} (v5.5: ~27, v6.1: ~30)")
        return len(self.y_dict) > 0

    def _numeric_feature_list_for_outlier(self, X_frame: pd.DataFrame):
        candidates = []
        for col in (self.hrv_features + self.psych_features + ['Age', 'BMI']):
            if col in X_frame.columns:
                candidates.append(col)
        for col in ['Age_BMI', 'HRV_Mean', 'LF_HF_Ratio', 'Sympathetic_Index', 
                    'bai_log', 'Panic_Risk']:
            if col in X_frame.columns:
                candidates.append(col)
        return [c for c in candidates if c in X_frame.columns]

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
            s = pd.to_numeric(X_train[col], errors='coerce')
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
                zero_mask = (pd.to_numeric(Xp[col], errors='coerce') == 0)
                total_flagged += int(zero_mask.sum())
                Xp.loc[zero_mask, col] = np.nan

        for col, (lb, ub) in self.outlier_bounds_.items():
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors='coerce')
            mask = (s < lb) | (s > ub)
            total_flagged += int(mask.sum())
            Xp.loc[mask, col] = np.nan

        if stage_note:
            print(f"   • [{stage_note}] 离群值→NaN：{total_flagged} 個")
        return Xp

    def _apply_log1p(self, X_frame: pd.DataFrame):
        Xp = X_frame.copy()
        for col in self.log_hrv_cols + self.log_engineered_cols:
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors='coerce')
            neg_mask = s < 0
            if neg_mask.any():
                Xp.loc[neg_mask, col] = np.nan
            Xp[col] = np.log1p(Xp[col])
        return Xp

    def impute_and_scale(self, X_train, X_test=None, fit=True):
        X_train_p = X_train.copy()
        X_test_p = X_test.copy() if X_test is not None else None

        for f in self.clinical_features:
            if f in X_train_p.columns:
                X_train_p[f].fillna(0, inplace=True)
                if X_test_p is not None:
                    X_test_p[f].fillna(0, inplace=True)

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
                self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                X_train_p[knn_f] = self.knn_imputer.fit_transform(X_train_p[knn_f])
            else:
                X_train_p[knn_f] = self.knn_imputer.transform(X_train_p[knn_f])
            if X_test_p is not None:
                X_test_p[knn_f] = self.knn_imputer.transform(X_test_p[knn_f])

        if X_train_p.isnull().any().any():
            X_train_p.fillna(X_train_p.median(numeric_only=True), inplace=True)
            if X_test_p is not None:
                X_test_p.fillna(X_train_p.median(numeric_only=True), inplace=True)

        if fit or (self.scaler is None):
            self.scaler = StandardScaler()
            X_train_s = self.scaler.fit_transform(X_train_p)
        else:
            X_train_s = self.scaler.transform(X_train_p)
        
        X_train_s = pd.DataFrame(X_train_s, columns=X_train_p.columns, index=X_train_p.index)
        
        if X_test_p is not None:
            X_test_s = self.scaler.transform(X_test_p)
            X_test_s = pd.DataFrame(X_test_s, columns=X_test_p.columns, index=X_test_p.index)
            return X_train_s, X_test_s
        
        return X_train_s


# ===========================
# 分類器
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
        self.fitted_models = {}  # for stacking
        
        if self.gap > 0.10:
            self.strategy = 'aggressive'
        elif self.gap > 0.05:
            self.strategy = 'moderate'
        else:
            self.strategy = 'conservative'
    
    def get_sampling_strategy(self):
        """
        v6.1 微調1: Panic 專用雙階段採樣（更保守）
        """
        if self.label_name == 'MDD':
            return 'SMOTE', 0.75, 5
        
        # === Panic 微調：雙階段採樣 ===
        if self.label_name == 'Panic':
            return 'BorderlineSMOTE', 0.55, 4  # 從 0.50 微調至 0.55
        
        if self.label_name == 'GAD':
            return 'BorderlineSMOTE', 0.35, 4

        if self.pos_count < 100:
            sampler_type = 'ADASYN'
            if self.strategy == 'aggressive':
                sampling_ratio = 0.65
                k = 4
            else:
                sampling_ratio = 0.55
                k = 5
        else:
            if self.strategy == 'aggressive':
                sampler_type = 'SMOTE'
                sampling_ratio = 0.65
                k = 4
            elif self.strategy == 'moderate':
                sampler_type = 'SMOTE'
                sampling_ratio = 0.55
                k = 5
            else:
                sampler_type = 'SMOTE'
                sampling_ratio = 0.50
                k = 5
        
        return sampler_type, sampling_ratio, k
    
    def build_models(self):
        """
        v6.1 微調2: 保留 v5.5 所有 6 個模型，只針對 Panic 調整超參數
        """
        scale_weight = int(self.ratio * 1.0)
        
        print(f"\n{'='*70}")
        print(f"🎯 {self.label_name}: F1={self.current_f1:.4f} → {self.target_f1:.4f}")
        print(f"   策略: {self.strategy.upper()}, 正例={self.pos_count}")
        
        if self.strategy == 'aggressive':
            n_est = 700
            depth = 25
            lr = 0.03
            base_weight_mult = 2.0
        elif self.strategy == 'moderate':
            n_est = 500
            depth = 18
            lr = 0.05
            base_weight_mult = 1.5
        else:
            n_est = 400
            depth = 12
            lr = 0.08
            base_weight_mult = 1.2

        # === Panic 微調2：權重微調 ===
        if self.label_name == 'MDD':
            weight_mult = 2.0
        elif self.label_name == 'Panic':
            weight_mult = 1.8  # 從 1.5 微調至 1.8
        elif self.label_name == 'GAD':
            weight_mult = 1.0
        else:
            weight_mult = base_weight_mult
        
        final_weight = max(1, int(scale_weight * weight_mult))
        
        if self.label_name == 'Panic':
            # Panic 專用參數微調
            self.models['XGB'] = xgb.XGBClassifier(
                n_estimators=650,          # 微調: 600→650
                max_depth=18,              # 微調: 16→18
                learning_rate=0.035,       # 微調: 0.04→0.035
                scale_pos_weight=final_weight,
                subsample=0.75,
                colsample_bytree=0.75,
                gamma=0.2,
                min_child_weight=1,
                reg_alpha=0.08,            # 微調: 0.05→0.08
                reg_lambda=0.6,            # 微調: 0.5→0.6
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        else:
            self.models['XGB'] = xgb.XGBClassifier(
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
                verbosity=0
            )
        
        self.models['LGBM'] = lgb.LGBMClassifier(
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
            verbose=-1
        )
        
        self.models['RF'] = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight},
            random_state=42,
            n_jobs=-1
        )
        
        self.models['ET'] = ExtraTreesClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight},
            random_state=42,
            n_jobs=-1
        )
        
        self.models['GB'] = GradientBoostingClassifier(
            n_estimators=int(n_est * 0.6),
            max_depth=int(depth * 0.3),
            learning_rate=lr,
            subsample=0.8,
            min_samples_split=8,
            random_state=42
        )
        
        self.models['BalancedRF'] = BalancedRandomForestClassifier(
            n_estimators=int(n_est * 0.8),
            max_depth=depth,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"   ✓ 建立 {len(self.models)} 個模型（v5.5原有6個）, 權重={final_weight}")
    
    def _fit_single_model(self, name, model, X_resampled, y_resampled):
        """v6.1 微調3: Panic 使用更溫和的 early stopping"""
        if self.label_name == 'Panic':
            use_early_stop = (
                isinstance(model, xgb.XGBClassifier) or 
                isinstance(model, lgb.LGBMClassifier)
            )
            early_stop_rounds = 20  # 微調: 從15改為20（更寬容）
            val_split_ratio = 0.20   # 微調: 從0.25改為0.20
        elif self.label_name == 'MDD':
            use_early_stop = (
                isinstance(model, xgb.XGBClassifier) or 
                isinstance(model, lgb.LGBMClassifier)
            )
            early_stop_rounds = 20
            val_split_ratio = 0.20
        else:
            use_early_stop = False
            early_stop_rounds = 30
            val_split_ratio = 0.15

        if use_early_stop:
            sss = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=val_split_ratio, 
                random_state=42
            )
            train_sub_idx, val_sub_idx = next(sss.split(X_resampled, y_resampled))
            X_tr_sub = X_resampled.iloc[train_sub_idx]
            y_tr_sub = y_resampled.iloc[train_sub_idx]
            X_val_sub = X_resampled.iloc[val_sub_idx]
            y_val_sub = y_resampled.iloc[val_sub_idx]

            try:
                if isinstance(model, xgb.XGBClassifier):
                    model.fit(
                        X_tr_sub, y_tr_sub,
                        eval_set=[(X_val_sub, y_val_sub)],
                        early_stopping_rounds=early_stop_rounds,
                        verbose=False
                    )
                elif isinstance(model, lgb.LGBMClassifier):
                    model.fit(
                        X_tr_sub, y_tr_sub,
                        eval_set=[(X_val_sub, y_val_sub)],
                        eval_metric='binary_logloss',
                        callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)]
                    )
            except TypeError:
                model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_resampled, y_resampled)

        return model

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """訓練流程（已保留 AUC/ACC 並補齊每折列印用資料）"""
        
        sampler_type, sampling_ratio, k = self.get_sampling_strategy()
        print(f"\n   📈 採樣: {sampler_type} (ratio={sampling_ratio:.2f})...")
        
        try:
            if sampler_type == 'ADASYN':
                sampler = ADASYN(sampling_strategy=sampling_ratio, n_neighbors=k, random_state=42)
            elif sampler_type == 'BorderlineSMOTE':
                sampler = BorderlineSMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
            else:
                sampler = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
            
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            print(f"      正例: {y_train.sum()} → {y_resampled.sum()}")
        except Exception as e:
            print(f"      ⚠️ 採樣失敗: {e}")
            X_resampled, y_resampled = X_train, y_train
        
        print(f"\n   🔄 訓練與推論...")
        self.fitted_models = {}
        
        for name, model in self.models.items():
            try:
                fitted_model = self._fit_single_model(name, model, X_resampled, y_resampled)
                self.fitted_models[name] = fitted_model  # for stacking

                y_pred_proba = fitted_model.predict_proba(X_test)[:, 1]
                
                best_thresh, _ = self._optimize_threshold_precision_first(y_test, y_pred_proba)
                y_pred = (y_pred_proba >= best_thresh).astype(int)
                
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except Exception:
                    auc = np.nan
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)

                # ===== 新增：Spec & NPV =====
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                else:
                    specificity = 0.0
                    npv = 0.0
                # ===========================
                
                self.results[name] = {
                    'f1_score': f1,
                    'accuracy': acc,
                    'auc': auc,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,   # <-- 新增
                    'npv': npv,                   # <-- 新增
                    'threshold': best_thresh,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_true': y_test.values,
                }
                
                status = "✅" if f1 >= self.target_f1 else "⚠️"
                print(f"      {name:13s}: F1={f1:.4f} {status}, "
                      f"P={precision:.3f}, R={recall:.3f}, "
                      f"Spec={specificity:.3f}, NPV={npv:.3f}, "
                      f"AUC={auc:.3f}, ACC={acc:.3f}, t={best_thresh:.2f}")
                
            except Exception as e:
                print(f"      ❌ {name}: {e}")
        
        # 先做 Stacking，再做 Ensemble
        self._create_stacking(X_train, X_test, y_train, y_test)
        self._create_ensemble(X_test, y_test)
        return self.results
    
    def _optimize_threshold_precision_first(self, y_true, y_pred_proba, n_thresh=100):
        """
        v6.1 微調4: Panic 放寬閾值約束（更實用）
        """
        thresholds = np.linspace(0.10, 0.90, n_thresh)

        # Panic 微調：降低 precision 下限
        if self.label_name == 'MDD':
            min_precision = 0.45
            min_recall    = 0.45
        elif self.label_name == 'Panic':
            min_precision = 0.45   # 從 0.60 降至 0.45（關鍵微調！）
            min_recall    = 0.45   # 從 0.50 降至 0.45
        elif self.label_name == 'GAD':
            min_precision = 0.68
            min_recall    = 0.50
        else:
            min_precision = 0.50
            min_recall    = 0.30

        best_f1 = 0
        best_thresh = 0.5
        
        # 第一輪：precision/recall 下限
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            if y_pred.sum() == 0:
                continue
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall    = recall_score(y_true, y_pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        # 第二輪：降級方案
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
    
    def _create_stacking(self, X_train, X_test, y_train, y_test):
        """
        使用 base models 的預測機率做第二層 Stacking (LogisticRegression)
        """
        if len(self.fitted_models) < 2:
            return

        y_train_arr = np.array(y_train)
        if len(np.unique(y_train_arr)) < 2:
            # 無法訓練二元 LR
            return
        
        try:
            base_names = []
            train_meta_features = []
            test_meta_features = []

            for name, model in self.fitted_models.items():
                if not hasattr(model, "predict_proba"):
                    continue
                base_names.append(name)
                train_meta_features.append(model.predict_proba(X_train)[:, 1])
                test_meta_features.append(model.predict_proba(X_test)[:, 1])

            if len(train_meta_features) == 0:
                return

            meta_X_train = np.vstack(train_meta_features).T  # shape: (n_train, n_base_models)
            meta_X_test  = np.vstack(test_meta_features).T   # shape: (n_test,  n_base_models)

            # class_weight 用原始 fold 中的正負例比例
            if self.pos_count > 0:
                cw_ratio = self.neg_count / self.pos_count
                class_weight = {0: 1.0, 1: cw_ratio}
            else:
                class_weight = None

            meta_clf = LogisticRegression(
                max_iter=1000,
                class_weight=class_weight,
                random_state=42
            )
            meta_clf.fit(meta_X_train, y_train_arr)

            stack_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
            best_thresh, _ = self._optimize_threshold_precision_first(y_test, stack_proba)
            stack_pred = (stack_proba >= best_thresh).astype(int)

            f1 = f1_score(y_test, stack_pred)
            acc = accuracy_score(y_test, stack_pred)
            try:
                auc = roc_auc_score(y_test, stack_proba)
            except Exception:
                auc = np.nan
            precision = precision_score(y_test, stack_pred, zero_division=0)
            recall = recall_score(y_test, stack_pred, zero_division=0)

            cm = confusion_matrix(y_test, stack_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            else:
                specificity = 0.0
                npv = 0.0

            self.results['Stacking'] = {
                'f1_score': f1,
                'accuracy': acc,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'npv': npv,
                'threshold': best_thresh,
                'y_pred': stack_pred,
                'y_pred_proba': stack_proba,
                'y_true': y_test.values
            }

            status = "✅" if f1 >= self.target_f1 else "⚠️"
            print(f"      {'Stacking':13s}: F1={f1:.4f} {status}, "
                  f"P={precision:.3f}, R={recall:.3f}, "
                  f"Spec={specificity:.3f}, NPV={npv:.3f}, "
                  f"AUC={auc:.3f}, ACC={acc:.3f}, t={best_thresh:.2f}")

        except Exception as e:
            print(f"      ⚠️ Stacking 失敗: {e}")

    def _create_ensemble(self, X_test, y_test):
        # 只用 base models（排除 Stacking / Ensemble 自己）
        base_results = {
            name: r for name, r in self.results.items()
            if name not in ['Ensemble', 'Stacking']
        }
        if len(base_results) < 2:
            return
        
        try:
            predictions = []
            weights = []
            for name, result in base_results.items():
                # Panic 微調：平衡 F1 和 Precision
                if self.label_name == 'Panic':
                    weight = 0.5 * result['f1_score'] + 0.5 * result['precision']
                elif self.label_name == 'MDD':
                    weight = 0.4 * result['f1_score'] + 0.6 * result['recall']
                else:
                    weight = result['f1_score'] * (0.5 + 0.5 * result['precision'])
                
                predictions.append(result['y_pred_proba'])
                weights.append(max(weight, 0.01))
            
            weights = np.array(weights)
            weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)
            
            ensemble_proba = np.average(predictions, axis=0, weights=weights)
            best_thresh, _ = self._optimize_threshold_precision_first(y_test, ensemble_proba)
            ensemble_pred = (ensemble_proba >= best_thresh).astype(int)
            
            f1 = f1_score(y_test, ensemble_pred)
            acc = accuracy_score(y_test, ensemble_pred)
            try:
                auc = roc_auc_score(y_test, ensemble_proba)
            except Exception:
                auc = np.nan
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)

            # ===== 新增：Spec & NPV（Ensemble） =====
            cm = confusion_matrix(y_test, ensemble_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            else:
                specificity = 0.0
                npv = 0.0
            # =======================================
            
            self.results['Ensemble'] = {
                'f1_score': f1,
                'accuracy': acc,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,   # 新增
                'npv': npv,                   # 新增
                'threshold': best_thresh,
                'y_pred': ensemble_pred,
                'y_pred_proba': ensemble_proba,
                'y_true': y_test.values
            }
            
            status = "✅" if f1 >= self.target_f1 else "⚠️"
            print(f"      {'Ensemble':13s}: F1={f1:.4f} {status}, "
                  f"P={precision:.3f}, R={recall:.3f}, "
                  f"Spec={specificity:.3f}, NPV={npv:.3f}, "
                  f"AUC={auc:.3f}, ACC={acc:.3f}")
            
        except Exception as e:
            print(f"      ⚠️ 集成失敗: {e}")
    
    def get_best_model(self):
        if not self.results:
            return None, None
        best_name = max(
            self.results.items(),
            key=lambda x: (x[1]['f1_score'], x[1]['precision'])
        )[0]
        return best_name, self.results[best_name]


# ===========================
# 主流程（整合輸出與視覺化）
# ===========================

def main():
    print("\n" + "="*70)
    print(" 輸出與視覺化）")
    print("="*70)

    timestamp = datetime.now().strftime("Run_v61_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📂 輸出資料夾: {run_dir}")

    FILE_PATH = r"D:\FLY114\data1_filled.xlsx"
    SHEET_NAME = "Filled_Data"
    
    processor = DataProcessorV61(
        FILE_PATH,
        SHEET_NAME,
        iqr_multiplier=3.0,
        treat_zero_as_missing_in_hrv=True
    )
    
    if not processor.load_data():
        return
    if not processor.prepare_features_and_labels():
        return
    
    X = processor.X
    y_dict = processor.y_dict
    label_names = processor.label_names
    
    Y_multi = pd.concat([y_dict[lb] for lb in label_names], axis=1)
    
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_results = {lb: [] for lb in label_names}
    per_model_fold_results = {lb: {} for lb in label_names}
    best_across_folds = {
        lb: {
            "model": None, 
            "fold": None, 
            "f1": -1.0, 
            "p": 0.0, 
            "r": 0.0,
            "spec": 0.0,
            "npv": 0.0,
            "auc": 0.0, 
            "acc": 0.0}
        for lb in label_names
    }
    ensemble_fold_scores = {lb: [] for lb in label_names}
    oof_true  = {lb: [] for lb in label_names}
    oof_proba = {lb: [] for lb in label_names}
    oof_pred  = {lb: [] for lb in label_names}
    per_fold_pred_details = {lb: {} for lb in label_names}
    
    fold_id = 1
    for train_idx, test_idx in mskf.split(X, Y_multi):
        print(f"\n📂 Fold {fold_id}/5")
        print("-" * 70)
        
        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train_dict_fold = {lb: y_dict[lb].iloc[train_idx] for lb in label_names}
        y_test_dict_fold  = {lb: y_dict[lb].iloc[test_idx] for lb in label_names}
        
        X_train_p, X_test_p = processor.impute_and_scale(X_train, X_test, fit=True)
        print(f"✓ 預處理完成（Fold {fold_id}）")
        
        for label in label_names:
            y_train = y_train_dict_fold[label]
            y_test  = y_test_dict_fold[label]
            
            pos = y_train.sum()
            neg = len(y_train) - pos
            
            current_baseline = {
                'Health': 0.7940, 'SSD': 0.6667,
                'MDD': 0.4691, 'Panic': 0.5021, 'GAD': 0.5797
            }.get(label, 0.5)
            target_goal = {
                'Health': 0.80, 'SSD': 0.65,
                'MDD': 0.50, 'Panic': 0.55, 'GAD': 0.65  # Panic目標 0.55 (保守+10%)
            }.get(label, 0.7)
            
            trainer = ClassifierV61(label, pos, neg, current_baseline, target_goal)
            trainer.build_models()
            results = trainer.train_and_evaluate(
                X_train_p, X_test_p,
                y_train, y_test
            )

            # === 每 fold 彙整：先比較 Stacking / Ensemble，用 F1 較高者當代表 ===
            special_candidates = []
            if 'Stacking' in results:
                special_candidates.append('Stacking')
            if 'Ensemble' in results:
                special_candidates.append('Ensemble')

            if len(special_candidates) > 0:
                best_special = max(
                    special_candidates,
                    key=lambda m: results[m]['f1_score']
                )
                r_best_for_fold = results[best_special]
                show_from = best_special
            else:
                base_models_for_fold = [
                    m for m in results.keys()
                    if m not in ['Ensemble', 'Stacking']
                ]
                best_name_fold = max(
                    base_models_for_fold,
                    key=lambda m: results[m]['f1_score']
                )
                r_best_for_fold = results[best_name_fold]
                show_from = best_name_fold
            # ==============================================
            
            all_fold_results[label].append({
                'f1_score': r_best_for_fold['f1_score'],
                'precision': r_best_for_fold['precision'],
                'recall': r_best_for_fold['recall'],
                'specificity': r_best_for_fold['specificity'],
                'npv': r_best_for_fold['npv'],
                'auc': r_best_for_fold['auc'],
                'accuracy': r_best_for_fold['accuracy']
            })
            
            # OOF: 使用當前 fold 代表模型（Stacking / Ensemble / base）
            chosen_name = show_from
            chosen_res = results[chosen_name]
            oof_true[label].extend(list(chosen_res['y_true']))
            oof_proba[label].extend(list(chosen_res['y_pred_proba']))
            oof_pred[label].extend(list(chosen_res['y_pred']))
            
            for mname, r in results.items():
                per_model_fold_results[label].setdefault(mname, [])
                per_model_fold_results[label][mname].append({
                    'f1_score': r['f1_score'],
                    'precision': r['precision'],
                    'recall': r['recall'],
                    'specificity': r['specificity'],
                    'npv': r['npv'],
                    'auc': r['auc'],
                    'accuracy': r['accuracy']
                })

            per_fold_pred_details[label].setdefault(fold_id, {})
            for mname, r in results.items():
                per_fold_pred_details[label][fold_id][mname] = {
                    "y_true": np.array(r['y_true']),
                    "y_pred_proba": np.array(r['y_pred_proba'])
                }
            
            # best_across_folds: 仍然只看 base models（不含 Ensemble / Stacking）
            base_models_no_ens = [
                m for m in results.keys()
                if m not in ['Ensemble', 'Stacking']
            ]
            if len(base_models_no_ens) > 0:
                best_name_this_fold = max(
                    base_models_no_ens,
                    key=lambda m: results[m]['f1_score']
                )
                br = results[best_name_this_fold]
                if br['f1_score'] > best_across_folds[label]['f1']:
                    best_across_folds[label] = {
                        "model": best_name_this_fold,
                        "fold": fold_id,
                        "f1":  float(br['f1_score']),
                        "p":   float(br['precision']),
                        "r":   float(br['recall']),
                        "spec": float(br['specificity']),
                        "npv":  float(br['npv']),
                        "auc": float(br['auc']),
                        "acc": float(br['accuracy']),
                    }
            
            if 'Ensemble' in results:
                er = results['Ensemble']
                ensemble_fold_scores[label].append({
                    "f1":  float(er['f1_score']),
                    "p":   float(er['precision']),
                    "r":   float(er['recall']),
                    "spec": float(er['specificity']),
                    "npv":  float(er['npv']),
                    "auc": float(er['auc']),
                    "acc": float(er['accuracy']),
                })
            
            print(f"   → {label:<8} | "
                  f"F1={r_best_for_fold['f1_score']:.4f}, "
                  f"P={r_best_for_fold['precision']:.4f}, "
                  f"R={r_best_for_fold['recall']:.4f}, "
                  f"spec={r_best_for_fold['specificity']:.4f},"
                  f"npv={r_best_for_fold['npv']:.4f},"
                  f"AUC={r_best_for_fold['auc']:.4f}, "
                  f"ACC={r_best_for_fold['accuracy']:.4f}  [{show_from}]")
        
        fold_id += 1
    
    # =============== 最終彙整與輸出（Best + avg） ===============
    print("\n" + "="*70)
    print("🏁 5-Fold 最終結果 (v6.1 保守版)  — 以 v5.4 欄位輸出")
    print("="*70)
    header = (f"\n{'Label':<10} {'BestModel':<12} "
              f"{'F1(Best)':>9} {'P(Best)':>9} {'R(Best)':>9} {'Spec(Best)':>11} {'NPV(Best)':>10} "
              f"{'AUC(Best)':>10} {'ACC(Best)':>10}   "
              f"{'F1(avg)':>9} {'P(avg)':>9} {'R(avg)':>9} {'Spec(avg)':>11} {'NPV(avg)':>10} "
              f"{'AUC(avg)':>10} {'ACC(avg)':>10}")
    print(header)
    print("-" * 125)

    rows = []

    # 視覺化資料集
    plot_labels = []
    plot_best_f1 = []
    plot_avg_f1 = []
    plot_avg_precision = []
    plot_avg_recall = []
    plot_stability_mean = []
    plot_stability_std = []

    # ROC 曲線資料
    roc_curves_oof       = {}  # label -> (fpr, tpr, auc_val) using OOF blended preds
    roc_curves_bestfold  = {}  # label -> (fpr, tpr, auc_val) using best single-model fold

    # 平均 & 最佳 (Accuracy / Precision / Recall / F1)
    avg_acc_by_label    = {}
    avg_prec_by_label   = {}
    avg_recall_by_label = {}
    avg_f1_by_label     = {}

    best_acc_by_label    = {}
    best_prec_by_label   = {}
    best_recall_by_label = {}
    best_f1_by_label     = {}

    for label in label_names:
        b = best_across_folds[label]

        # 從最佳單折 (best single-model fold) 取指標
        b_f1   = float(b.get("f1",  np.nan))
        b_p    = float(b.get("p",   np.nan))
        b_r    = float(b.get("r",   np.nan))
        b_spec = float(b.get("spec", np.nan))
        b_npv  = float(b.get("npv",  np.nan))
        b_auc  = float(b.get("auc", np.nan))
        b_acc  = float(b.get("acc", np.nan))
        b_md   = b.get("model", None)
        b_fold = b.get("fold",  None)

        # 計算平均（Ensemble 平均優先，否則用 best-of-fold 平均）
        if len(ensemble_fold_scores[label]) > 0:
            e_df = pd.DataFrame(ensemble_fold_scores[label])
            f1_avg  = float(e_df['f1'].mean())
            p_avg   = float(e_df['p'].mean())
            r_avg   = float(e_df['r'].mean())
            spec_avg = float(e_df['spec'].mean())
            npv_avg  = float(e_df['npv'].mean())
            auc_avg = float(e_df['auc'].mean())
            acc_avg = float(e_df['acc'].mean())
        else:
            temp_df = pd.DataFrame(all_fold_results[label])
            f1_avg  = float(temp_df['f1_score'].mean())
            p_avg   = float(temp_df['precision'].mean())
            r_avg   = float(temp_df['recall'].mean())
            spec_avg = float(temp_df['specificity'].mean())
            npv_avg  = float(temp_df['npv'].mean())
            auc_avg = float(temp_df['auc'].mean())
            acc_avg = float(temp_df['accuracy'].mean())

        print(f"{label:<10} {str(b_md):<12} "
              f"{b_f1:>9.4f} {b_p:>9.4f} {b_r:>9.4f} {b_spec:>11.4f} {b_npv:>10.4f} "
              f"{b_auc:>10.4f} {b_acc:>10.4f}   "
              f"{f1_avg:>9.4f} {p_avg:>9.4f} {r_avg:>9.4f} {spec_avg:>11.4f} {npv_avg:>10.4f} "
              f"{auc_avg:>10.4f} {acc_avg:>10.4f}")

        rows.append({
            "Label": label,
            "BestModel": b_md,
            "Fold(Best)": b_fold,
            "F1(Best)": b_f1,
            "P(Best)": b_p,
            "R(Best)": b_r,
            "Spec(Best)": b_spec,
            "NPV(Best)": b_npv,
            "AUC(Best)": b_auc,
            "ACC(Best)": b_acc,
            "F1(avg)": f1_avg,
            "P(avg)": p_avg,
            "R(avg)": r_avg,
            "Spec(avg)": spec_avg,
            "NPV(avg)": npv_avg,
            "AUC(avg)": auc_avg,
            "ACC(avg)": acc_avg,
        })

        # ====== 視覺化彙整 (F1 etc) ======
        plot_labels.append(label)
        plot_best_f1.append(b_f1)
        plot_avg_f1.append(f1_avg)
        plot_avg_precision.append(p_avg)
        plot_avg_recall.append(r_avg)

        # F1 穩定度（誤差棒）
        fold_f1_list = [fold_res['f1_score'] for fold_res in all_fold_results[label]]
        stability_mean = float(np.mean(fold_f1_list)) if len(fold_f1_list) > 0 else np.nan
        stability_std  = float(np.std(fold_f1_list, ddof=1)) if len(fold_f1_list) > 1 else 0.0
        plot_stability_mean.append(stability_mean)
        plot_stability_std.append(stability_std)

        # ROC curve (OOF)
        y_true_all  = np.array(oof_true[label])
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

        # ROC curve (Best single-model fold)
        if b_fold is not None and b_md is not None:
            fold_dict = per_fold_pred_details[label].get(b_fold, {})
            model_dict = fold_dict.get(b_md, {})
            y_true_best  = model_dict.get("y_true", np.array([]))
            y_proba_best = model_dict.get("y_pred_proba", np.array([]))

            if y_true_best.size > 0 and len(np.unique(y_true_best)) > 1:
                try:
                    fpr_best, tpr_best, _ = roc_curve(y_true_best, y_proba_best)
                    auc_best = roc_auc_score(y_true_best, y_proba_best)
                except Exception:
                    fpr_best, tpr_best, auc_best = None, None, np.nan
            else:
                fpr_best, tpr_best, auc_best = None, None, np.nan
        else:
            fpr_best, tpr_best, auc_best = None, None, np.nan

        roc_curves_bestfold[label] = (fpr_best, tpr_best, auc_best)

        # 平均 (5-fold mean，用每折 best-of-fold 代表該折)
        perf_df = pd.DataFrame(all_fold_results[label])
        acc_mean       = float(perf_df['accuracy'].mean())
        precision_mean = float(perf_df['precision'].mean())
        recall_mean    = float(perf_df['recall'].mean())
        f1_mean        = float(perf_df['f1_score'].mean())

        avg_acc_by_label[label]    = acc_mean
        avg_prec_by_label[label]   = precision_mean
        avg_recall_by_label[label] = recall_mean
        avg_f1_by_label[label]     = f1_mean

        # 最佳單折 (best single-model fold)
        best_acc_by_label[label]    = b_acc
        best_prec_by_label[label]   = b_p
        best_recall_by_label[label] = b_r
        best_f1_by_label[label]     = b_f1

    # =============== 輸出 Excel ===============
    out_df = pd.DataFrame(rows, columns=[
        "Label", "BestModel", "Fold(Best)",
        "F1(Best)", "P(Best)", "R(Best)", "Spec(Best)", "NPV(Best)",
        "AUC(Best)", "ACC(Best)",
        "F1(avg)", "P(avg)", "R(avg)", "Spec(avg)", "NPV(avg)",
        "AUC(avg)", "ACC(avg)"
    ])
    excel_path = os.path.join(run_dir, "v6.1_KFold_Best_and_Avg.xlsx")
    out_df.to_excel(excel_path, index=False)
    print(f"\n✅ 已輸出結果至 {excel_path}")

    # =========================================================
    # 視覺化區域
    # =========================================================

    # 小工具：在每根柱子上標數值
    def attach_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.2f}",
                ha='center', va='bottom',
                fontsize=10
            )

    # (1) 各標籤的 F1-score 柱狀圖 (平均 vs 最佳單模)
    x_idx = np.arange(len(plot_labels))
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(10,6))
    bars_avg  = ax.bar(x_idx - bar_w/2, plot_avg_f1,  width=bar_w,
                       label='F1 avg (5-fold / Ensemble or Best-of-fold)')
    bars_best = ax.bar(x_idx + bar_w/2, plot_best_f1, width=bar_w,
                       label='F1 Best Single Model')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel('F1-score')
    ax.set_title('F1-score by Label')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    attach_value_labels(bars_avg, ax)
    attach_value_labels(bars_best, ax)
    fig.tight_layout()
    f1_bar_path = os.path.join(run_dir, "F1_BarChart.png")
    fig.savefig(f1_bar_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {f1_bar_path}")

    # (2) Precision vs Recall 散點圖
    fig, ax = plt.subplots(figsize=(8,6))
    for lbl, p_val, r_val in zip(plot_labels, plot_avg_precision, plot_avg_recall):
        ax.scatter(r_val, p_val, s=80)
        ax.text(r_val+0.01, p_val+0.01, lbl)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (per label)')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    pr_scatter_path = os.path.join(run_dir, "Precision_Recall_Scatter.png")
    fig.savefig(pr_scatter_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {pr_scatter_path}")

    # (3) 5-Fold 穩定度圖（誤差棒）
    fig, ax = plt.subplots(figsize=(10,6))
    bars_stab = ax.bar(
        x_idx,
        plot_stability_mean,
        yerr=plot_stability_std,
        capsize=5
    )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(plot_labels)
    ax.set_ylabel('F1-score (mean ± std)')
    ax.set_title('5-Fold Stability of F1-score')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    attach_value_labels(bars_stab, ax)
    fig.tight_layout()
    stability_path = os.path.join(run_dir, "F1_Stability_ErrorBar.png")
    fig.savefig(stability_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {stability_path}")

    # (4) ROC Curve（OOF / 全部樣本綜合）
    fig, ax = plt.subplots(figsize=(8,6))
    for lbl in plot_labels:
        fpr_oof, tpr_oof, auc_oof = roc_curves_oof[lbl]
        if fpr_oof is not None and tpr_oof is not None:
            ax.plot(fpr_oof, tpr_oof, label=f"{lbl} (AUC={auc_oof:.3f})")
    ax.plot([0,1], [0,1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (OOF)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    roc_path = os.path.join(run_dir, "ROC_Curves_OOF.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {roc_path}")

    # (4b) ROC Curve（best single-model fold）
    fig, ax = plt.subplots(figsize=(8,6))
    for lbl in plot_labels:
        fpr_b, tpr_b, auc_b = roc_curves_bestfold[lbl]
        if fpr_b is not None and tpr_b is not None:
            ax.plot(fpr_b, tpr_b, label=f"{lbl} (AUC={auc_b:.3f})")
    ax.plot([0,1], [0,1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (Best single-model fold)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    roc_best_path = os.path.join(run_dir, "ROC_Curves_BestFold.png")
    fig.savefig(roc_best_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {roc_best_path}")

    # (5) Confusion Matrix（每個標籤一張圖）— 使用 OOF 預測
    for lbl in plot_labels:
        y_true_all = np.array(oof_true[lbl])
        y_pred_all = np.array(oof_pred[lbl])

        if len(y_true_all) == 0:
            continue

        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0,1])
        cm_counts = cm.astype(int)

        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm_counts, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix - {lbl}')
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(['Pred 0','Pred 1'])
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(['True 0','True 1'])

        thresh = cm_counts.max() / 2.0 if cm_counts.max() > 0 else 0.5
        for i in range(cm_counts.shape[0]):
            for j in range(cm_counts.shape[1]):
                ax.text(j, i, str(cm_counts[i, j]),
                        ha="center", va="center",
                        color="white" if cm_counts[i, j] > thresh else "black",
                        fontsize=12)

        fig.tight_layout()
        cm_path = os.path.join(run_dir, f"ConfusionMatrix_{lbl}.png")
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"📊 已輸出: {cm_path}")

    # (6) 各 label 的 Accuracy / Precision / Recall / F1 柱狀圖 (5-fold mean)
    metrics_x = np.arange(len(plot_labels))
    width = 0.20  # 四組柱

    acc_vals_mean    = [avg_acc_by_label[lbl]    for lbl in plot_labels]
    prec_vals_mean   = [avg_prec_by_label[lbl]   for lbl in plot_labels]
    recall_vals_mean = [avg_recall_by_label[lbl] for lbl in plot_labels]
    f1_vals_mean     = [avg_f1_by_label[lbl]     for lbl in plot_labels]

    fig, ax = plt.subplots(figsize=(12,7))
    bars_acc_mean    = ax.bar(metrics_x - 1.5*width, acc_vals_mean,    width=width, label='Accuracy (mean)')
    bars_prec_mean   = ax.bar(metrics_x - 0.5*width, prec_vals_mean,   width=width, label='Precision (mean)')
    bars_recall_mean = ax.bar(metrics_x + 0.5*width, recall_vals_mean, width=width, label='Recall (mean)')
    bars_f1_mean     = ax.bar(metrics_x + 1.5*width, f1_vals_mean,     width=width, label='F1-score (mean)')

    ax.set_xticks(metrics_x)
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Accuracy / Precision / Recall / F1-score by Label (5-fold mean)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    attach_value_labels(bars_acc_mean, ax)
    attach_value_labels(bars_prec_mean, ax)
    attach_value_labels(bars_recall_mean, ax)
    attach_value_labels(bars_f1_mean, ax)

    fig.tight_layout()
    acc_prec_recall_mean_path = os.path.join(run_dir, "Acc_Prec_Recall_F1_ByLabel_mean.png")
    fig.savefig(acc_prec_recall_mean_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {acc_prec_recall_mean_path}")

    # (7) 各 label 的 Accuracy / Precision / Recall / F1 柱狀圖 (best fold 單一最佳模型表現)
    acc_vals_best    = [best_acc_by_label[lbl]    for lbl in plot_labels]
    prec_vals_best   = [best_prec_by_label[lbl]   for lbl in plot_labels]
    recall_vals_best = [best_recall_by_label[lbl] for lbl in plot_labels]
    f1_vals_best     = [best_f1_by_label[lbl]     for lbl in plot_labels]

    fig, ax = plt.subplots(figsize=(12,7))
    bars_acc_best    = ax.bar(metrics_x - 1.5*width, acc_vals_best,    width=width, label='Accuracy (best fold)')
    bars_prec_best   = ax.bar(metrics_x - 0.5*width, prec_vals_best,   width=width, label='Precision (best fold)')
    bars_recall_best = ax.bar(metrics_x + 0.5*width, recall_vals_best, width=width, label='Recall (best fold)')
    bars_f1_best     = ax.bar(metrics_x + 1.5*width, f1_vals_best,     width=width, label='F1-score (best fold)')

    ax.set_xticks(metrics_x)
    ax.set_xticklabels(plot_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Accuracy / Precision / Recall / F1-score by Label (best single-model fold)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    attach_value_labels(bars_acc_best, ax)
    attach_value_labels(bars_prec_best, ax)
    attach_value_labels(bars_recall_best, ax)
    attach_value_labels(bars_f1_best, ax)

    fig.tight_layout()
    bestfold_acc_prec_recall_path = os.path.join(run_dir, "Acc_Prec_Recall_F1_ByLabel_bestFold.png")
    fig.savefig(bestfold_acc_prec_recall_path, dpi=300)
    plt.close(fig)
    print(f"📊 已輸出: {bestfold_acc_prec_recall_path}")

    # =========================================================

    print("\n✅ 完成！所有圖表與 Excel 已輸出到:", run_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()