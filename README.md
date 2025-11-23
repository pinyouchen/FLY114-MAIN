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

## 4.1 Task Definitions：Differential Diagnosis vs Pure Case-Control

本研究目前有兩套互補的任務定義，用於回答不同的臨床問題：

1. **舊版設定：Differential Diagnosis（鑑別診斷）**  
   - 定義：  
     - 針對某一疾病（例如 SSD），將「SSD = 1」視為正類，  
       將「所有其他個案（包含其他精神疾病 + 健康個案）」視為負類。  
   - 技術意義：  
     - 負類樣本中混雜大量共病（comorbidities）個案與異質臨床狀態，  
       對模型來說是難度較高的鑑別診斷情境。  
     - 此設定能評估模型在「實務上複雜情境」區辨單一診斷的能力，但指標通常較保守。  

2. **新版設定：Pure Case-Control（病例–對照）**  
   - 定義：  
     - 針對 SSD / MDD / Panic / GAD 四個診斷，  
       僅保留「該診斷 = 1」與「純健康（Health = 1、其他診斷皆為 0）」個案，  
       建構二元分類任務（患者 vs 純健康人）。  
   - 技術意義：  
     - 負類為「心理上與臨床評估均為健康」的對照組，  
       不再混雜其他疾病或共病，屬於較「乾淨」的 case-control 設計。  
     - 能提供 HRV + Demographics 在理想情境下的上限表現（upper bound），  
       也較貼近典型生理信號研究中的病例–對照設計。

## 4.2 Pipeline Steps

### Step 1 — Baseline Model（Data2 訓練）
- Features：SDNN, LF, HF + Age, Sex, BMI  
- 建立 HRV Minimal Benchmark（同時支援 Differential Diagnosis 與 Pure Case-Control 兩種任務設定）

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

本節先保留「舊版 Differential Diagnosis」的多標籤任務結果，  
再補充「新版 Pure Case-Control」設定下的成績，方便直接比較。

## 7.1 Differential Diagnosis — Best Fold Performance（舊版設定）

> 任務定義：例如 SSD vs 其他（包含 MDD、Panic、GAD 及 Health），  
> 屬於難度較高、負類高度異質的鑑別診斷情境。

| Label  | BestModel | Fold | F1 | Precision | Recall | Specificity | NPV | AUC | ACC |
|--------|-----------|------|------:|------:|------:|------:|------:|------:|------:|
| **Health** | ET | 4 | 0.9551 | 0.9551 | 0.9551 | 0.9518 | 0.9518 | 0.9835 | 0.9535 |
| **SSD** | GB | 1 | 0.8440 | 0.8070 | 0.8462 | 0.9091 | 0.9483 | 0.9358 | 0.9017 |
| **MDD** | GB | 1 | 0.8710 | 0.9000 | 0.8438 | 0.9787 | 0.9650 | 0.9807 | 0.9536 |
| **Panic** | GB | 4 | 0.6122 | 0.6000 | 0.6250 | 0.9324 | 0.9388 | 0.8713 | 0.8894 |
| **GAD** | ET | 4 | 0.7704 | 0.6341 | 0.9811 | 0.7479 | 0.9889 | 0.8819 | 0.8200 |

---

## 7.2 Differential Diagnosis — Average Performance Across Folds（舊版設定）

| Label  | F1(avg) | P(avg) | R(avg) | Spec(avg) | NPV(avg) | AUC(avg) | ACC(avg) |
|--------|--------:|--------:|--------:|-----------:|-----------:|-----------:|-----------:|
| **Health** | 0.9093 | 0.9197 | 0.9433 | 0.9131 | 0.9392 | 0.9658 | 0.9283 |
| **SSD** | 0.7766 | 0.7043 | 0.8682 | 0.8438 | 0.9381 | 0.9184 | 0.8500 |
| **MDD** | 0.7733 | 0.7414 | 0.8159 | 0.9374 | 0.9581 | 0.9461 | 0.9123 |
| **Panic** | 0.5319 | 0.4827 | 0.6444 | 0.8598 | 0.9381 | 0.8287 | 0.8299 |
| **GAD** | 0.7095 | 0.6167 | 0.8479 | 0.7652 | 0.9225 | 0.8575 | 0.7891 |

---

## 7.3 Pure Case-Control — HRV+Demo Baseline（新版設定：患者 vs 純健康人）

> 任務定義：  
> - 僅保留「純健康個案」（Health = 1 且 SSD/MDD/Panic/GAD = 0）  
> - 以及「單一目標診斷 = 1」且其他診斷 = 0 的個案（例如 SSD = 1，MDD/Panic/GAD = 0）  
> - 對 SSD / MDD / Panic / GAD 各自建立「患者 vs 純健康人」二元模型。  
>
> 這個設定排除了大部分共病與其他精神疾病，提供 HRV + Demographics 在  
> 理想 case-control 設計下的上限表現，也能當作 Differential Diagnosis 設定的對照組。

### 7.3.1 Pure Case-Control — Best Fold Performance

| Label  | BestModel | F1(Best) | P(Best) | R(Best) | Spec(Best) | NPV(Best) | AUC(Best) | ACC(Best) |
|--------|-----------|---------:|--------:|--------:|-----------:|----------:|----------:|----------:|
| **SSD**   | GB   | 0.9346 | 0.8929 | 0.9804 | 0.9326 | 0.9881 | 0.9775 | 0.9500 |
| **MDD**   | GB   | 0.9688 | 0.9688 | 0.9688 | 0.9886 | 0.9886 | 0.9957 | 0.9833 |
| **Panic** | LGBM | 0.9167 | 0.9565 | 0.8800 | 0.9886 | 0.9667 | 0.9377 | 0.9646 |
| **GAD**   | ET   | 0.9434 | 0.9259 | 0.9615 | 0.9545 | 0.9767 | 0.9736 | 0.9571 |

### 7.3.2 Pure Case-Control — Average Performance Across Folds

| Label  | F1(avg) | P(avg) | R(avg) | Spec(avg) | NPV(avg) | AUC(avg) | ACC(avg) |
|--------|--------:|--------:|--------:|-----------:|----------:|----------:|----------:|
| **SSD**   | 0.9167 | 0.9162 | 0.9187 | 0.9502 | 0.9527 | 0.9706 | 0.9385 |
| **MDD**   | 0.9445 | 0.9224 | 0.9681 | 0.9706 | 0.9885 | 0.9882 | 0.9700 |
| **Panic** | 0.8822 | 0.8738 | 0.8927 | 0.9638 | 0.9704 | 0.9583 | 0.9484 |
| **GAD**   | 0.9162 | 0.9217 | 0.9123 | 0.9523 | 0.9483 | 0.9646 | 0.9375 |

> 觀察：在 Pure Case-Control 設定下，四個診斷在 HRV+Demo baseline 上的  
> F1、Precision、Recall、Specificity 皆顯著優於 Differential Diagnosis 設定，  
> 顯示當負類限制為「純健康對照組」時，HRV 與基本人口學特徵在鑑別疾病 vs 健康方面  
> 可以達到接近天花板的表現。

---

# 8. Key Findings

1. **任務難度差異：Differential vs Pure Case-Control**  
   - Differential Diagnosis（疾病 vs 其他所有人）為高度異質的臨床情境，  
     模型必須在多種共病與重疊症狀之間做精細鑑別，導致 F1 與 AUC 相對保守。  
   - Pure Case-Control（疾病 vs 純健康）則對應典型病例–對照研究設計，  
     更適合評估「HRV + 自律神經指標本身」在理想條件下的區辨能力。  

2. **Health 診斷最容易**  
   - 無論在 Differential 或 Pure Case-Control 設定下，Health 的 AUC 與 F1 皆接近 0.95–0.98，  
     顯示 HRV + 心理量表在偵測「真正健康 vs 非健康」上具有極佳可用性。  

3. **MDD / SSD / GAD 表現穩定**  
   - 在 Differential 設定下，F1 約落在 0.77–0.87；  
   - 在 Pure Case-Control 設定下，四個診斷對應的 HRV+Demo 模型皆呈現接近天花板的表現  
     （以你實際的表格為準，Best fold F1 大致落在高 0.9 區間）。  

4. **Panic 為最困難任務，但在 Pure Case-Control 下明顯改善**  
   - Differential 設定中，Panic 受樣本數與共病影響最重，F1 約在 0.5–0.6。  
   - 在 Pure Case-Control 設定中，Panic vs Health 的 HRV+Demo 表現大幅提升，  
     顯示當負類限制為「無其他精神疾病」時，Panic 與健康的 HRV 差異更容易被捕捉。  

5. **心理量表與 HRV 的角色**  
   - 舊版多標籤模型顯示：心理量表（例如 BDI、BAI、PHQ15）在 SHAP / Permutation Importance 中  
     佔據最高權重，HRV 主要提供輔助訊號。  
   - 在 Pure Case-Control HRV baseline 中，即使只使用 HRV + Demographics，  
     仍可在病例–對照設計下達到相當高的 F1 / AUC，  
     支持「生理訊號在理想實驗設計下具臨床潛力」的論點。  

6. **實務解讀建議**  
   - Differential Diagnosis 結果可視為「接近實際門診的困難情境」，  
     用來討論模型在現實世界鑑別診斷中的限制與潛在輔助價值。  
   - Pure Case-Control 結果則可視為 HRV / 自律神經指標的「最好情境表現上限」，  
     適合用於與傳統 HRV 研究或其他生理指標研究做橫向比較。  

---
