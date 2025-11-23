# FLY114-MAIN  
臺大醫院臨床 HRV / 自律神經 / 心理量表多標籤分類研究

本專案整合臺大醫院兩組臨床資料（Data1 / Data2），建置多標籤（multi-label）分類模型，評估 HRV、自律神經與心理量表對五大診斷（Health / SSD / MDD / Panic / GAD）的區辨能力，並比較不同特徵組的模型效能，同時檢驗外部驗證（external validation）以確保跨資料集泛化能力。

---

# 1. Dataset Summary

| Dataset | N  | Purpose |
|--------|----|---------|
| **Data1** | 502 | External validation / Testing |
| **Data2** | 364 | Model development / Training |
| **Full Data** | 866 | Comprehensive feature modeling |

**Labels（Multi-label）**  
- Health  
- SSD  
- MDD  
- Panic  
- GAD  

---

# 2. Feature Groups

### 2.1 Baseline HRV
- HRV: SDNN, LF, HF  
- Demographics: Age, Sex, BMI  

### 2.2 Extended HRV
- All HRV features  
- HRV + Demographics  
- HRV + Demographics + Psychological Scales  

### 2.3 Psychological Scales Only
- phq15, bdi, bai, others  
- Demographics  

### 2.4 Full Feature Set
- Clinical  
- Autonomic  
- HRV  
- Psychological  
- Demographics  

---

# 3. Preprocessing Workflow

1. **資料整合與欄位統一**（比對 Data1 / Data2 重疊與非重疊特徵）  
2. **缺失值補值（兩種方法比較）**  
   - KNN Imputation  
   - Random Forest Regression (RFR) Imputation  
   - PCA 顯示 RFR 補值後類別分群較佳，因此後續主要採 RFR  
3. **IQR 離群值處理**  
4. **標準化（StandardScaler）**  
5. **特徵工程：心理量表加總、HRV 衍生比值等**  

---

# 4. Modeling Pipeline

整體建模流程如下：

### Step 1 — Baseline Model（Data2 訓練）
- Features：SDNN, LF, HF + Age, Sex, BMI  
- 建立 HRV Minimal Benchmark

### Step 2 — External Validation（Data1）
- 使用 Step 1 訓練的模型  
- 不重新訓練，直接測試 Data1  
- 評估跨資料集泛化能力

### Step 3 — Psychological Scales Only
- 僅使用心理量表  
- 目的為比較「生理 vs 心理」特徵的重要性

### Step 4 — Extended HRV
- All HRV  
- HRV + Demographics  
- HRV + Demographics + Scales  
- 評估 HRV 擴增是否提升診斷能力

### Step 5 — Full Feature Model（Data1 + Data2）
- 全特徵建模  
- 目標為取得最佳整體模型績效

---

# 5. Cross-Validation / Resampling / Models

### 5.1 CV Strategy
- Train/Test Split = 80 / 20  
- K-Fold Cross Validation on training split  

### 5.2 Oversampling（Label-Specific）
- Health：SMOTE  
- SSD：SMOTE  
- MDD：SMOTE  
- GAD：SMOTE  
- Panic：Borderline-SMOTE（最佳）

### 5.3 Machine Learning Models
- XGBoost  
- LightGBM  
- Gradient Boosting (GB)  
- Extra Trees (ET)  
- Random Forest  
- Balanced Random Forest  
- **Ensemble（最佳）**：加權平均後重新評估  

---

# 6. Evaluation Metrics

使用以下臨床級指標：
- **F1-score**  
- **Precision (P)**  
- **Recall (R)**  
- **Specificity (Spec)**  
- **Negative Predictive Value (NPV)**  
- **AUC (ROC)**  
- **Accuracy (ACC)**  

同時輸出：
- Confusion Matrix  
- ROC Curves  
- SHAP Value  
- Permutation Importance  

---

# 7. Final Performance (Per Label)

以下為你提供的最終完整績效表格（Best Fold 與平均值 Avg 全部收錄）。

## 7.1 Best Fold Performance

| Label  | BestModel | Fold | F1 | Precision | Recall | Specificity | NPV | AUC | ACC |
|--------|-----------|------|------:|------:|------:|------:|------:|------:|------:|
| **Health** | ET | 4 | 0.9551 | 0.9551 | 0.9551 | 0.9518 | 0.9518 | 0.9835 | 0.9535 |
| **SSD** | GB | 1 | 0.8440 | 0.8070 | 0.8462 | 0.9091 | 0.9483 | 0.9358 | 0.9017 |
| **MDD** | GB | 1 | 0.8710 | 0.9000 | 0.8438 | 0.9787 | 0.9650 | 0.9807 | 0.9536 |
| **Panic** | GB | 4 | 0.6122 | 0.6000 | 0.6250 | 0.9324 | 0.9388 | 0.8713 | 0.8894 |
| **GAD** | ET | 4 | 0.7704 | 0.6341 | 0.9811 | 0.7479 | 0.9889 | 0.8819 | 0.8200 |

---

## 7.2 Average Performance Across Folds

| Label  | F1(avg) | P(avg) | R(avg) | Spec(avg) | NPV(avg) | AUC(avg) | ACC(avg) |
|--------|--------:|--------:|--------:|-----------:|-----------:|-----------:|-----------:|
| **Health** | 0.9093 | 0.9197 | 0.9433 | 0.9131 | 0.9392 | 0.9658 | 0.9283 |
| **SSD** | 0.7766 | 0.7043 | 0.8682 | 0.8438 | 0.9381 | 0.9184 | 0.8500 |
| **MDD** | 0.7733 | 0.7414 | 0.8159 | 0.9374 | 0.9581 | 0.9461 | 0.9123 |
| **Panic** | 0.5319 | 0.4827 | 0.6444 | 0.8598 | 0.9381 | 0.8287 | 0.8299 |
| **GAD** | 0.7095 | 0.6167 | 0.8479 | 0.7652 | 0.9225 | 0.8575 | 0.7891 |

---

# 8. Key Findings

1. **Health** 診斷最容易、模型可超過 0.95（F1 / AUC 均接近完美）  
2. **MDD** 擁有最高的 AUC（0.98+），心理量表貢獻度最高  
3. **SSD / GAD** 表現穩定（F1 0.77–0.84），但仍受共病影響  
4. **Panic** 最困難（F1 ≈ 0.53–0.61），Recall 與資料分佈影響明顯  
5. 心理量表（BDI、BAI、PHQ15）在模型中擁有最高 SHAP / Importance  
6. HRV 單獨區辨力有限，但與量表合併後能提升 MDD / SSD 的辨識  

---
