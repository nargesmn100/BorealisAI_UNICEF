# 🌍 UNICEF × RBC Borealis AI Project

## AI-Powered Reconstruction of Multidimensional Child Deprivation

---

## 🧠 Project Overview

This project focuses on building a **machine learning system** to reconstruct **fine-grained spatial distributions of multidimensional child deprivation** using **coarse administrative-level official statistics** and **high-resolution proxy data**.

The system is designed to support **disaster forecasting and humanitarian decision-making**, where **timely and spatially precise insights** are critical for identifying vulnerable populations—especially children.

---

## 🎯 Core Problem

Humanitarian systems (e.g., UNICEF’s estimation of *Children in Need (CHIN)*) rely on combining:

- Hazard exposure (high-resolution)
- Population data (high-resolution)
- Poverty and deprivation data (**low-resolution, admin-level only**)

### ❗ Key Issue: Resolution Mismatch

- Hazard forecasts → fine-grained (e.g., 25km or better)
- Population data → very fine (e.g., 100m)
- Deprivation data → coarse (admin-level aggregates)

> This prevents accurate identification of **which specific communities are most vulnerable within a region**.

---

## ⚠️ Limitations of Current Approach

Current workaround uses:

- Relative Wealth Index (RWI)
- Settlement classification (urban/rural)
- Manual redistribution of poverty

### Problems:

- Assumes wealth ≈ deprivation (not always true)
- Does **not model multidimensional deprivation**
- Simply **reshapes aggregates**, rather than learning structure
- Limited ability to generalize across countries

---

## 🧩 Project Objective

Determine whether machine learning models can **reconstruct meaningful spatial patterns of multidimensional child deprivation** from coarse data using proxy variables—and identify **when they truly add information beyond redistribution baselines**.

---

## 📦 System Output

The system produces:

### 1. Grid-Level Predictions

- Fine-resolution estimates of deprivation across space

### 2. Multidimensional Deprivation Map

- Composite index derived from:
  - WASH deprivation
  - Health access deprivation
  - Education access deprivation

### 3. Uncertainty-Aware Outputs

- Confidence or uncertainty estimates per grid cell

### 4. Comparative Evaluation

- Performance vs baseline (RWI redistribution)

---

## 🏗️ System Design Philosophy

This is **not a pure prediction system**.

Instead, it is a:

> **Constraint-aware spatial inference system under data scarcity**

Key idea:

- Learn **relative spatial patterns** within regions
- Then enforce **consistency with official statistics**

---

## ⚙️ Constraint Strategy

### ✅ Post-Processing with Hard Administrative Reconciliation

Pipeline:

1. Model predicts fine-grained spatial variation
2. Predictions are **rescaled within each admin region**
3. Final outputs:
   - Preserve official totals exactly
   - Maintain spatial differentiation

---

## 🧪 Modeling Strategy

### Primary Targets

Predict each deprivation dimension separately:

- WASH
- Health
- Education

Then compute:

- Composite multidimensional deprivation index

---

## 📊 Data Structure

### Inputs

- Administrative-level deprivation indicators
- High-resolution proxy data:
  - Relative Wealth Index (RWI)
  - Population density / child population
  - Settlement type (urban/rural)
  - Accessibility metrics (healthcare, cities, schools)
  - Built-up area / night lights (optional)

### Output Resolution

- Grid-level predictions aligned to proxy data resolution

---

## 🧪 Experimental Framing

### Goal:

Evaluate **when ML adds value**

### Comparison Groups:

#### Baselines

- Uniform redistribution
- RWI-based redistribution

#### Models

- Interpretable models (regression, GAMs)
- Nonlinear models (gradient boosting, neural nets)

---

## 📏 Evaluation Criteria

### 1. Performance vs Baseline

- Does the model outperform RWI redistribution?

### 2. Structural Fidelity

- Does it preserve relationships between deprivation dimensions?

### 3. Spatial Accuracy

- Can it reconstruct fine-scale variation?

### 4. Uncertainty Calibration

- Are confidence estimates meaningful?

---

## ⚠️ Key Constraints

### Data Constraints

- Missing / inconsistent deprivation data
- Coarse administrative resolution
- Heterogeneous data sources

### Modeling Constraints

- Must preserve official totals
- Limited fine-resolution ground truth

### Operational Constraints

- Outputs must be:
  - Interpretable
  - Defensible
  - Suitable for prioritization (not official stats)

### Ethical Constraints

- Avoid overclaiming precision
- Maintain transparency in uncertainty
- Do not present outputs as official statistics

---

## ⚖️ Assumptions

- Proxy variables encode meaningful signals about deprivation
- Relationships between proxies and deprivation are learnable
- Within-region variation exists and can be inferred

---

## ❗ Key Risks

### False Precision

High-resolution outputs may appear more accurate than they are

### Proxy Bias

Proxies like RWI may not fully capture deprivation

### Generalization Failure

Models may not transfer well across regions

### Overfitting

Model may learn admin-level patterns instead of true spatial variation

### No Added Value

ML may not outperform simple redistribution

---

## 🧠 Success Criteria

The project is successful if it demonstrates:

- Improvement over RWI baseline
- Meaningful uncertainty-aware outputs
- Strong system design and reasoning
- Honest identification of limitations
- Clear humanitarian impact

---

## 🧩 System Scope

This is a full system including:

- Data ingestion
- Feature engineering
- Modeling
- Constraint reconciliation
- Output generation
- Evaluation framework

---

## 🚀 Positioning Statement

This project does not aim to replace official statistics.

Instead, it provides a **data-driven bridging methodology** to support **faster, more precise humanitarian decision-making under uncertainty**.
