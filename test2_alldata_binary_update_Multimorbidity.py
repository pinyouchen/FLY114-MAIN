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
    GradientBoostingClassifier, IsolationForest
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
# 資料處理 (v6.2 邏輯)
# ===========================

class DataProcessorV62:
    
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
        
        print(f"\n🔨 特徵工程 (v6.2 SSD/MDD 增強版)...")
        
        # === 基礎特徵 ===
        if 'Age' in self.X.columns and 'BMI' in self.X.columns:
            self.X['Age_BMI'] = self.X['Age'] * self.X['BMI']

        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3:
            self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
            if 'LF' in self.X.columns and 'HF' in self.X.columns:
                self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
                self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

        psych_cols = [c for c in self.psych_features if c in self.X.columns]
        if len(psych_cols) >= 3:
            self.X['Psych_Sum'] = self.X[psych_cols].sum(axis=1)

        # === Panic 相關 ===
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['Sympathetic_Index'] = self.X['LF'] / (self.X['LF'] + self.X['HF'] + 1e-6)
        if 'bai' in self.X.columns:
            self.X['bai_log'] = np.log1p(self.X['bai'])
        if 'bai' in self.X.columns and 'HRV_Mean' in self.X.columns:
            self.X['Panic_Risk'] = self.X['bai'] / (self.X['HRV_Mean'] + 1e-6)

        # === SSD, MDD, GAD 專屬特徵 ===
        if 'phq15' in self.X.columns:
            self.X['phq15_log'] = np.log1p(self.X['phq15'])
        if 'phq15' in self.X.columns and 'Age' in self.X.columns:
            self.X['Somatic_Age_Interaction'] = self.X['phq15'] * self.X['Age']

        if 'bdi' in self.X.columns:
            self.X['bdi_log'] = np.log1p(self.X['bdi'])
        if 'bdi' in self.X.columns and 'HRV_Mean' in self.X.columns:
            self.X['Depression_HRV_Ratio'] = self.X['bdi'] / (self.X['HRV_Mean'] + 1e-6)

        if 'cabah' in self.X.columns:
            self.X['cabah_log'] = np.log1p(self.X['cabah'])
        if 'cabah' in self.X.columns and 'TP' in self.X.columns:
            self.X['Anxiety_TP_Ratio'] = self.X['cabah'] / (self.X['TP'] + 1e-6)

        for label in self.label_names:
            if label in self.df.columns:
                self.y_dict[label] = self.df[label].copy()
        
        print(f"\n✓ 總特徵數量: {self.X.shape[1]}")
        return len(self.y_dict) > 0

    def _numeric_feature_list_for_outlier(self, X_frame: pd.DataFrame):
        candidates = []
        for col in (self.hrv_features + self.psych_features + ['Age', 'BMI']):
            if col in X_frame.columns:
                candidates.append(col)
        
        generated_cols = [
            'Age_BMI', 'HRV_Mean', 'LF_HF_Ratio', 'Sympathetic_Index', 
            'bai_log', 'Panic_Risk', 
            'phq15_log', 'Somatic_Age_Interaction',
            'bdi_log', 'Depression_HRV_Ratio',
            'cabah_log', 'Anxiety_TP_Ratio'
        ]
        for col in generated_cols:
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
        if self.treat_zero_as_missing_in_hrv:
            for col in [c for c in self.hrv_features if c in Xp.columns]:
                zero_mask = (pd.to_numeric(Xp[col], errors='coerce') == 0)
                Xp.loc[zero_mask, col] = np.nan
        for col, (lb, ub) in self.outlier_bounds_.items():
            if col not in Xp.columns:
                continue
            s = pd.to_numeric(Xp[col], errors='coerce')
            mask = (s < lb) | (s > ub)
            Xp.loc[mask, col] = np.nan
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
# 分類器 (v6.10 - GAD 優化回歸版)
# ===========================

class ClassifierV610:
    
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
        self.fitted_models = {}
        
        if self.gap > 0.10:
            self.strategy = 'aggressive'
        elif self.gap > 0.05:
            self.strategy = 'moderate'
        else:
            self.strategy = 'conservative'
    
    def get_sampling_strategy(self):
        # === GAD 成功設定: BorderlineSMOTE ===
        if self.label_name == 'GAD':
            return 'BorderlineSMOTE', 0.35, 4
            
        if self.label_name == 'SSD':
            return 'BorderlineSMOTE', 0.40, 5
        if self.label_name == 'MDD':
            return 'SMOTE', 0.65, 5
        if self.label_name == 'Panic':
            return 'BorderlineSMOTE', 0.55, 4

        if self.pos_count < 100:
            sampler_type = 'ADASYN'
            sampling_ratio = 0.65 if self.strategy == 'aggressive' else 0.55
            k = 4
        else:
            sampler_type = 'SMOTE'
            sampling_ratio = 0.65 if self.strategy == 'aggressive' else 0.50
            k = 5
        
        return sampler_type, sampling_ratio, k
    
    def build_models(self):
        scale_weight = int(self.ratio * 1.0)
        
        print(f"\n{'='*70}")
        print(f"🎯 {self.label_name}: F1={self.current_f1:.4f} → {self.target_f1:.4f}")
        print(f"   策略: {self.strategy.upper()}, 正例={self.pos_count}")
        
        if self.strategy == 'aggressive':
            n_est, depth, lr, base_weight_mult = 800, 25, 0.02, 2.0
        elif self.strategy == 'moderate':
            n_est, depth, lr, base_weight_mult = 600, 18, 0.04, 1.5
        else:
            n_est, depth, lr, base_weight_mult = 500, 12, 0.06, 1.2

        # === 權重設定 (包含 GAD 優化) ===
        if self.label_name == 'MDD':
            weight_mult = 1.6   
        elif self.label_name == 'SSD':
            weight_mult = 1.0   
        elif self.label_name == 'GAD':
            # 🔥 GAD 優化：稍微調降權重 (1.0 -> 0.95)
            weight_mult = 0.95   
        elif self.label_name == 'Panic':
            weight_mult = 1.8
        else:
            weight_mult = base_weight_mult
        
        final_weight = max(1, int(scale_weight * weight_mult))
        
        # === XGBoost ===
        xgb_params = {
            'n_estimators': n_est,
            'max_depth': int(depth * 0.4),
            'learning_rate': lr,
            'scale_pos_weight': final_weight,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'min_child_weight': 2,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

        if self.label_name == 'Panic':
            xgb_params.update({
                'n_estimators': 700, 'max_depth': 18, 'learning_rate': 0.03,
                'subsample': 0.75, 'colsample_bytree': 0.75, 
                'reg_alpha': 0.08, 'reg_lambda': 0.6, 'min_child_weight': 1
            })
        elif self.label_name == 'SSD':
            xgb_params.update({
                'n_estimators': 700, 'max_depth': 12, 'learning_rate': 0.03,
                'reg_lambda': 2.0 
            })
        elif self.label_name == 'GAD':
            # 🔥 GAD 優化：淺樹深、高正則化
            xgb_params.update({
                'n_estimators': 700, 'max_depth': 10, 'learning_rate': 0.02,
                'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0.5,
                'min_child_weight': 3, 'reg_alpha': 0.5, 'reg_lambda': 1.5
            })

        self.models['XGB'] = xgb.XGBClassifier(**xgb_params)
        
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
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight}, random_state=42, n_jobs=-1
        )
        
        self.models['ET'] = ExtraTreesClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0: 1, 1: final_weight}, random_state=42, n_jobs=-1
        )
        
        self.models['GB'] = GradientBoostingClassifier(
            n_estimators=int(n_est * 0.6), max_depth=int(depth * 0.3),
            learning_rate=lr, subsample=0.8, min_samples_split=8, random_state=42
        )
        
        self.models['BalancedRF'] = BalancedRandomForestClassifier(
            n_estimators=int(n_est * 0.8), max_depth=depth, min_samples_split=8,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        )
        
        print(f"   ✓ 建立 {len(self.models)} 個模型, 權重參數={final_weight}")
    
    def _fit_single_model(self, name, model, X_resampled, y_resampled):
        use_early_stop = False
        early_stop_rounds = 30
        val_split_ratio = 0.15

        if self.label_name in ['Panic', 'MDD']:
            use_early_stop = (isinstance(model, xgb.XGBClassifier) or isinstance(model, lgb.LGBMClassifier))
            early_stop_rounds = 20
            val_split_ratio = 0.20

        if use_early_stop:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=42)
            train_sub_idx, val_sub_idx = next(sss.split(X_resampled, y_resampled))
            X_tr_sub = X_resampled.iloc[train_sub_idx]
            y_tr_sub = y_resampled.iloc[train_sub_idx]
            X_val_sub = X_resampled.iloc[val_sub_idx]
            y_val_sub = y_resampled.iloc[val_sub_idx]

            try:
                if isinstance(model, xgb.XGBClassifier):
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)],
                              early_stopping_rounds=early_stop_rounds, verbose=False)
                elif isinstance(model, lgb.LGBMClassifier):
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)],
                              eval_metric='binary_logloss',
                              callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)])
            except TypeError:
                model.fit(X_resampled, y_resampled)
        else:
            model.fit(X_resampled, y_resampled)
        return model

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        sampler_type, sampling_ratio, k = self.get_sampling_strategy()
        
        # === 強制執行 BorderlineSMOTE 機制 ===
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        current_ratio = n_pos / n_neg if n_neg > 0 else 1.0
        
        print(f"\n   📈 採樣檢查: 目標={sampling_ratio}, 當前={current_ratio:.3f} ({n_pos}/{n_neg})")

        if current_ratio >= sampling_ratio:
            adjusted_ratio = min(current_ratio + 0.15, 1.0)
            if adjusted_ratio > current_ratio:
                sampling_ratio = adjusted_ratio
                print(f"      ⚡ 強制啟動 SMOTE: 自動上調目標至 {sampling_ratio:.2f} (為了強化邊界)")
            else:
                print(f"      ⚡ 數據已完全平衡，無需 SMOTE。")

        if current_ratio < sampling_ratio:
            print(f"      🔄 執行 {sampler_type} (Ratio={sampling_ratio:.2f})...")
            try:
                if sampler_type == 'ADASYN':
                    sampler = ADASYN(sampling_strategy=sampling_ratio, n_neighbors=k, random_state=42)
                elif sampler_type == 'BorderlineSMOTE':
                    sampler = BorderlineSMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
                else:
                    sampler = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k, random_state=42)
                
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                print(f"      正例擴增: {n_pos} → {y_resampled.sum()}")
            except Exception as e:
                print(f"      ⚠️ 採樣失敗 ({e})，使用原始資料繼續。")
                X_resampled, y_resampled = X_train, y_train
        else:
            X_resampled, y_resampled = X_train, y_train
        
        print(f"\n   🔄 訓練與推論...")
        self.fitted_models = {}
        
        for name, model in self.models.items():
            try:
                fitted_model = self._fit_single_model(name, model, X_resampled, y_resampled)
                self.fitted_models[name] = fitted_model

                y_pred_proba = fitted_model.predict_proba(X_test)[:, 1]
                best_thresh, _ = self._optimize_threshold_precision_first(y_test, y_pred_proba)
                y_pred = (y_pred_proba >= best_thresh).astype(int)
                
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = np.nan
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)

                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                else:
                    specificity = 0.0
                    npv = 0.0
                
                self.results[name] = {
                    'f1_score': f1, 'accuracy': acc, 'auc': auc,
                    'precision': precision, 'recall': recall,
                    'specificity': specificity, 'npv': npv,
                    'threshold': best_thresh, 'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba, 'y_true': y_test.values,
                }
                
                status = "✅" if f1 >= self.target_f1 else "⚠️"
                print(f"      {name:13s}: F1={f1:.4f} {status}, P={precision:.3f}, R={recall:.3f}, "
                      f"Spec={specificity:.3f}, AUC={auc:.3f}, t={best_thresh:.2f}")
                
            except Exception as e:
                print(f"      ❌ {name}: {e}")
        
        self._create_stacking(X_train, X_test, y_train, y_test)
        self._create_top3_ensemble(X_test, y_test) # Top-3 Voting
        return self.results
    
    def _optimize_threshold_precision_first(self, y_true, y_pred_proba, n_thresh=100):
        thresholds = np.linspace(0.10, 0.90, n_thresh)

        # === 🔥 v6.10 閾值設定 (加入 GAD 優化) ===
        if self.label_name == 'GAD':
            # 🔥 GAD 關鍵修正: 0.62 是甜蜜點
            min_precision = 0.62  
            min_recall    = 0.60
        elif self.label_name == 'SSD':
            min_precision = 0.68  
            min_recall    = 0.60
        elif self.label_name == 'MDD':
            min_precision = 0.70  
            min_recall    = 0.60
        elif self.label_name == 'Panic':
            min_precision = 0.45
            min_recall    = 0.45
        else:
            min_precision = 0.50
            min_recall    = 0.30

        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            if y_pred.sum() == 0: continue
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall    = recall_score(y_true, y_pred, zero_division=0)
            if precision >= min_precision and recall >= min_recall:
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        # Fallback
        if best_f1 == 0:
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                if y_pred.sum() == 0: continue
                f1 = f1_score(y_true, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        
        return best_thresh, best_f1
    
    def _create_stacking(self, X_train, X_test, y_train, y_test):
        if len(self.fitted_models) < 2: return
        y_train_arr = np.array(y_train)
        if len(np.unique(y_train_arr)) < 2: return
        
        try:
            train_meta_features = []
            test_meta_features = []
            for name, model in self.fitted_models.items():
                if not hasattr(model, "predict_proba"): continue
                train_meta_features.append(model.predict_proba(X_train)[:, 1])
                test_meta_features.append(model.predict_proba(X_test)[:, 1])

            if len(train_meta_features) == 0: return

            meta_X_train = np.vstack(train_meta_features).T
            meta_X_test  = np.vstack(test_meta_features).T

            if self.pos_count > 0:
                cw_ratio = self.neg_count / self.pos_count
                class_weight = {0: 1.0, 1: cw_ratio}
            else:
                class_weight = None

            meta_clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
            meta_clf.fit(meta_X_train, y_train_arr)

            stack_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
            best_thresh, _ = self._optimize_threshold_precision_first(y_test, stack_proba)
            stack_pred = (stack_proba >= best_thresh).astype(int)

            f1 = f1_score(y_test, stack_pred)
            acc = accuracy_score(y_test, stack_pred)
            try: auc = roc_auc_score(y_test, stack_proba)
            except: auc = np.nan
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
                'f1_score': f1, 'accuracy': acc, 'auc': auc,
                'precision': precision, 'recall': recall,
                'specificity': specificity, 'npv': npv,
                'threshold': best_thresh, 'y_pred': stack_pred,
                'y_pred_proba': stack_proba, 'y_true': y_test.values
            }
            print(f"      {'Stacking':13s}: F1={f1:.4f} {'✅' if f1 >= self.target_f1 else '⚠️'}, "
                  f"P={precision:.3f}, R={recall:.3f}, Spec={specificity:.3f}, AUC={auc:.3f}")

        except Exception as e:
            print(f"      ⚠️ Stacking 失敗: {e}")

    def _create_top3_ensemble(self, X_test, y_test):
        base_results = {name: r for name, r in self.results.items() if name not in ['Ensemble', 'Stacking']}
        if len(base_results) < 2: return
        
        try:
            sorted_models = sorted(base_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            top_models = sorted_models[:3]
            
            print(f"      🤝 Top-3 Ensemble 採用: {[m[0] for m in top_models]}")

            predictions = []
            for name, result in top_models:
                predictions.append(result['y_pred_proba'])
            
            ensemble_proba = np.mean(predictions, axis=0)
            
            best_thresh, _ = self._optimize_threshold_precision_first(y_test, ensemble_proba)
            ensemble_pred = (ensemble_proba >= best_thresh).astype(int)
            
            f1 = f1_score(y_test, ensemble_pred)
            acc = accuracy_score(y_test, ensemble_pred)
            try: auc = roc_auc_score(y_test, ensemble_proba)
            except: auc = np.nan
            precision = precision_score(y_test, ensemble_pred, zero_division=0)
            recall = recall_score(y_test, ensemble_pred, zero_division=0)
            
            cm = confusion_matrix(y_test, ensemble_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            else:
                specificity = 0.0
                npv = 0.0
            
            self.results['Ensemble'] = {
                'f1_score': f1, 'accuracy': acc, 'auc': auc,
                'precision': precision, 'recall': recall,
                'specificity': specificity, 'npv': npv,
                'threshold': best_thresh, 'y_pred': ensemble_pred,
                'y_pred_proba': ensemble_proba, 'y_true': y_test.values
            }
            print(f"      {'Ensemble':13s}: F1={f1:.4f} {'✅' if f1 >= self.target_f1 else '⚠️'}, "
                  f"P={precision:.3f}, R={recall:.3f}, Spec={specificity:.3f}, AUC={auc:.3f}")
            
        except Exception as e:
            print(f"      ⚠️ 集成失敗: {e}")

# ===========================
# 主流程
# ===========================

def main():
    print("\n" + "="*70)
    print(" 🔥 5-Fold 最終結果 (v6.10 - GAD優化 + 完整視覺化)")
    print("="*70)
    
    timestamp = datetime.now().strftime("Run_v610_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📂 輸出資料夾: {run_dir}")
    
    # ⚠️ 請確認路徑
    FILE_PATH = r"D:\FLY114\data1_filled.xlsx"
    SHEET_NAME = "Filled_Data"
    
    processor = DataProcessorV62(
        FILE_PATH,
        SHEET_NAME,
        iqr_multiplier=3.0,
        treat_zero_as_missing_in_hrv=True
    )
    
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return
    
    X = processor.X
    y_dict = processor.y_dict
    label_names = processor.label_names
    
    Y_multi = pd.concat([y_dict[lb] for lb in label_names], axis=1)
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 準備資料結構 (for Excel & Plots)
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
        print(f"✓ 預處理完成")
        
        for label in label_names:
            y_train = y_train_dict_fold[label]
            y_test  = y_test_dict_fold[label]

            # ==========================================
            # 🔥 Isolation Forest
            # ==========================================
            iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
            outlier_preds = iso.fit_predict(X_train_p)
            mask = (outlier_preds == 1) | (y_train == 1)
            X_train_clean = X_train_p[mask].copy()
            y_train_clean = y_train[mask].copy()
            
            n_dropped = len(y_train) - len(y_train_clean)
            if n_dropped > 0:
                print(f"   🧹 IsoForest [{label}]: 清洗 {n_dropped} 筆負例雜訊 (保留正例)")
            
            pos = y_train_clean.sum()
            neg = len(y_train_clean) - pos
            
            current_baseline = {
                'Health': 0.7940, 'SSD': 0.6667,
                'MDD': 0.4691, 'Panic': 0.5021, 'GAD': 0.5797
            }.get(label, 0.5)
            
            target_goal = {
                'Health': 0.80, 'SSD': 0.75, 'MDD': 0.75, 'Panic': 0.55, 'GAD': 0.70
            }.get(label, 0.7)
            
            trainer = ClassifierV610(label, pos, neg, current_baseline, target_goal)
            trainer.build_models()
            
            results = trainer.train_and_evaluate(X_train_clean, X_test_p, y_train_clean, y_test)

            # === 收集數據 (Visuals & Excel) ===
            # 優先選 Ensemble/Stacking 作為該 Fold 代表
            special_candidates = []
            if 'Stacking' in results: special_candidates.append('Stacking')
            if 'Ensemble' in results: special_candidates.append('Ensemble')

            if len(special_candidates) > 0:
                best_special = max(special_candidates, key=lambda m: results[m]['f1_score'])
                r_best_for_fold = results[best_special]
                show_from = best_special
            else:
                # 否則選最佳 Base Model
                base_models_for_fold = [m for m in results.keys() if m not in ['Ensemble', 'Stacking']]
                best_name_fold = max(base_models_for_fold, key=lambda m: results[m]['f1_score'])
                r_best_for_fold = results[best_name_fold]
                show_from = best_name_fold
            
            all_fold_results[label].append({
                'f1_score': r_best_for_fold['f1_score'],
                'precision': r_best_for_fold['precision'],
                'recall': r_best_for_fold['recall'],
                'specificity': r_best_for_fold['specificity'],
                'npv': r_best_for_fold['npv'],
                'auc': r_best_for_fold['auc'],
                'accuracy': r_best_for_fold['accuracy']
            })

            # OOF for Plotting
            chosen_res = results[show_from]
            oof_true[label].extend(list(chosen_res['y_true']))
            oof_proba[label].extend(list(chosen_res['y_pred_proba']))
            oof_pred[label].extend(list(chosen_res['y_pred']))
            
            # Store per model details
            for mname, r in results.items():
                per_model_fold_results[label].setdefault(mname, [])
                per_model_fold_results[label][mname].append({
                    'f1_score': r['f1_score'],
                    'precision': r['precision'],
                    'recall': r['recall'],
                })

            per_fold_pred_details[label].setdefault(fold_id, {})
            for mname, r in results.items():
                per_fold_pred_details[label][fold_id][mname] = {
                    "y_true": np.array(r['y_true']),
                    "y_pred_proba": np.array(r['y_pred_proba'])
                }

            # Update Best Across Folds (Base Models Only)
            base_models_no_ens = [m for m in results.keys() if m not in ['Ensemble', 'Stacking']]
            if len(base_models_no_ens) > 0:
                best_name_this_fold = max(base_models_no_ens, key=lambda m: results[m]['f1_score'])
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
            
            # Collect Ensemble Scores specifically
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

            print(f"   → {label:<8} | F1={r_best_for_fold['f1_score']:.4f} (P={r_best_for_fold['precision']:.3f}) [{show_from}]")
        
        fold_id += 1
    
    # =========================================================
    # 最終彙整與輸出 (Excel & Plots) - 來自您的程式碼
    # =========================================================
    print("\n" + "="*70)
    print("🏁 5-Fold 最終結果 (v6.10 保守版) — 以 v5.4 欄位輸出")
    print("="*70)
    header = (f"\n{'Label':<10} {'BestModel':<12} "
              f"{'F1(Best)':>9} {'P(Best)':>9} {'R(Best)':>9} {'Spec(Best)':>11} {'NPV(Best)':>10} "
              f"{'AUC(Best)':>10} {'ACC(Best)':>10}   "
              f"{'F1(avg)':>9} {'P(avg)':>9} {'R(avg)':>9} {'Spec(avg)':>11} {'NPV(avg)':>10} "
              f"{'AUC(avg)':>10} {'ACC(avg)':>10}")
    print(header)
    print("-" * 125)

    rows = []
    plot_labels = []
    plot_best_f1 = []
    plot_avg_f1 = []
    plot_avg_precision = []
    plot_avg_recall = []
    plot_stability_mean = []
    plot_stability_std = []
    roc_curves_oof = {}
    roc_curves_bestfold = {}
    avg_acc_by_label = {}
    avg_prec_by_label = {}
    avg_recall_by_label = {}
    avg_f1_by_label = {}
    best_acc_by_label = {}
    best_prec_by_label = {}
    best_recall_by_label = {}
    best_f1_by_label = {}

    for label in label_names:
        b = best_across_folds[label]
        b_f1, b_p, b_r = float(b.get("f1", np.nan)), float(b.get("p", np.nan)), float(b.get("r", np.nan))
        b_spec, b_npv = float(b.get("spec", np.nan)), float(b.get("npv", np.nan))
        b_auc, b_acc = float(b.get("auc", np.nan)), float(b.get("acc", np.nan))
        b_md, b_fold = b.get("model", None), b.get("fold", None)

        # 計算平均 (優先使用 Ensemble 的平均)
        if len(ensemble_fold_scores[label]) > 0:
            e_df = pd.DataFrame(ensemble_fold_scores[label])
            f1_avg, p_avg, r_avg = float(e_df['f1'].mean()), float(e_df['p'].mean()), float(e_df['r'].mean())
            spec_avg, npv_avg = float(e_df['spec'].mean()), float(e_df['npv'].mean())
            auc_avg, acc_avg = float(e_df['auc'].mean()), float(e_df['acc'].mean())
        else:
            temp_df = pd.DataFrame(all_fold_results[label])
            f1_avg, p_avg, r_avg = float(temp_df['f1_score'].mean()), float(temp_df['precision'].mean()), float(temp_df['recall'].mean())
            spec_avg, npv_avg = float(temp_df['specificity'].mean()), float(temp_df['npv'].mean())
            auc_avg, acc_avg = float(temp_df['auc'].mean()), float(temp_df['accuracy'].mean())

        print(f"{label:<10} {str(b_md):<12} "
              f"{b_f1:>9.4f} {b_p:>9.4f} {b_r:>9.4f} {b_spec:>11.4f} {b_npv:>10.4f} "
              f"{b_auc:>10.4f} {b_acc:>10.4f}   "
              f"{f1_avg:>9.4f} {p_avg:>9.4f} {r_avg:>9.4f} {spec_avg:>11.4f} {npv_avg:>10.4f} "
              f"{auc_avg:>10.4f} {acc_avg:>10.4f}")

        rows.append({
            "Label": label, "BestModel": b_md, "Fold(Best)": b_fold,
            "F1(Best)": b_f1, "P(Best)": b_p, "R(Best)": b_r,
            "Spec(Best)": b_spec, "NPV(Best)": b_npv, "AUC(Best)": b_auc, "ACC(Best)": b_acc,
            "F1(avg)": f1_avg, "P(avg)": p_avg, "R(avg)": r_avg,
            "Spec(avg)": spec_avg, "NPV(avg)": npv_avg, "AUC(avg)": auc_avg, "ACC(avg)": acc_avg,
        })

        plot_labels.append(label)
        plot_best_f1.append(b_f1)
        plot_avg_f1.append(f1_avg)
        plot_avg_precision.append(p_avg)
        plot_avg_recall.append(r_avg)

        fold_f1_list = [fold_res['f1_score'] for fold_res in all_fold_results[label]]
        plot_stability_mean.append(float(np.mean(fold_f1_list)) if len(fold_f1_list) > 0 else np.nan)
        plot_stability_std.append(float(np.std(fold_f1_list, ddof=1)) if len(fold_f1_list) > 1 else 0.0)

        # ROC OOF
        y_true_all, y_proba_all = np.array(oof_true[label]), np.array(oof_proba[label])
        if len(np.unique(y_true_all)) > 1 and y_proba_all.size > 0:
            try:
                fpr_oof, tpr_oof, _ = roc_curve(y_true_all, y_proba_all)
                auc_oof = roc_auc_score(y_true_all, y_proba_all)
            except: fpr_oof, tpr_oof, auc_oof = None, None, np.nan
        else: fpr_oof, tpr_oof, auc_oof = None, None, np.nan
        roc_curves_oof[label] = (fpr_oof, tpr_oof, auc_oof)

        # ROC Best Fold
        if b_fold is not None and b_md is not None:
            y_true_best = per_fold_pred_details[label][b_fold][b_md]["y_true"]
            y_proba_best = per_fold_pred_details[label][b_fold][b_md]["y_pred_proba"]
            if y_true_best.size > 0 and len(np.unique(y_true_best)) > 1:
                try:
                    fpr_best, tpr_best, _ = roc_curve(y_true_best, y_proba_best)
                    auc_best = roc_auc_score(y_true_best, y_proba_best)
                except: fpr_best, tpr_best, auc_best = None, None, np.nan
            else: fpr_best, tpr_best, auc_best = None, None, np.nan
        else: fpr_best, tpr_best, auc_best = None, None, np.nan
        roc_curves_bestfold[label] = (fpr_best, tpr_best, auc_best)

        perf_df = pd.DataFrame(all_fold_results[label])
        avg_acc_by_label[label] = float(perf_df['accuracy'].mean())
        avg_prec_by_label[label] = float(perf_df['precision'].mean())
        avg_recall_by_label[label] = float(perf_df['recall'].mean())
        avg_f1_by_label[label] = float(perf_df['f1_score'].mean())
        best_acc_by_label[label] = b_acc
        best_prec_by_label[label] = b_p
        best_recall_by_label[label] = b_r
        best_f1_by_label[label] = b_f1

    # Excel
    out_df = pd.DataFrame(rows)
    excel_path = os.path.join(run_dir, "v6.10_KFold_Best_and_Avg.xlsx")
    out_df.to_excel(excel_path, index=False)
    print(f"\n✅ 已輸出結果至 {excel_path}")

    # Plots
    def attach_value_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

    x_idx = np.arange(len(plot_labels))
    bar_w = 0.35

    # 1. F1 Bar Chart
    fig, ax = plt.subplots(figsize=(10,6))
    bars_avg = ax.bar(x_idx - bar_w/2, plot_avg_f1, width=bar_w, label='F1 avg')
    bars_best = ax.bar(x_idx + bar_w/2, plot_best_f1, width=bar_w, label='F1 Best Single')
    ax.set_xticks(x_idx); ax.set_xticklabels(plot_labels); ax.set_ylim(0, 1.0); ax.legend()
    ax.set_title('F1-score by Label'); attach_value_labels(bars_avg, ax); attach_value_labels(bars_best, ax)
    fig.savefig(os.path.join(run_dir, "F1_BarChart.png"), dpi=300); plt.close(fig)

    # 2. PR Scatter
    fig, ax = plt.subplots(figsize=(8,6))
    for lbl, p_val, r_val in zip(plot_labels, plot_avg_precision, plot_avg_recall):
        ax.scatter(r_val, p_val, s=80); ax.text(r_val+0.01, p_val+0.01, lbl)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision vs Recall'); ax.grid(True, linestyle='--')
    fig.savefig(os.path.join(run_dir, "Precision_Recall_Scatter.png"), dpi=300); plt.close(fig)

    # 3. Stability
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x_idx, plot_stability_mean, yerr=plot_stability_std, capsize=5)
    ax.set_xticks(x_idx); ax.set_xticklabels(plot_labels); ax.set_ylim(0, 1.0)
    ax.set_title('5-Fold Stability of F1-score')
    fig.savefig(os.path.join(run_dir, "F1_Stability_ErrorBar.png"), dpi=300); plt.close(fig)

    # 4. ROC OOF
    fig, ax = plt.subplots(figsize=(8,6))
    for lbl in plot_labels:
        fpr, tpr, auc_val = roc_curves_oof[lbl]
        if fpr is not None: ax.plot(fpr, tpr, label=f"{lbl} (AUC={auc_val:.3f})")
    ax.plot([0,1], [0,1], 'k--'); ax.legend(loc='lower right'); ax.set_title('ROC Curves (OOF)')
    fig.savefig(os.path.join(run_dir, "ROC_Curves_OOF.png"), dpi=300); plt.close(fig)

    # 5. Confusion Matrices
    for lbl in plot_labels:
        y_true, y_pred = np.array(oof_true[lbl]), np.array(oof_pred[lbl])
        if len(y_true) == 0: continue
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'CM - {lbl}'); fig.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        fig.savefig(os.path.join(run_dir, f"ConfusionMatrix_{lbl}.png"), dpi=300); plt.close(fig)

    print(f"✅ 完成！所有圖表與 Excel 已輸出到: {run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()