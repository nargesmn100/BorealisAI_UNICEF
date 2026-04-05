# UNICEF × RBC Borealis AI Project
## Problem Statement, Approach, and Evaluation Framework

---

## Problem Statement

### Context (what space?)
This project sits in the space of:

- humanitarian forecasting
- disaster preparedness
- child vulnerability estimation
- geospatial machine learning
- official statistics under data scarcity

More specifically, it focuses on the UNICEF-style challenge of estimating where vulnerable children are located before or during climate-related emergencies, when decision-makers need spatially precise information for prioritization but official child poverty or deprivation data are only available at coarse administrative levels.

---

### Gap (what’s missing today?)
Today, there is a major mismatch between the spatial resolution of available datasets:

- hazard and exposure data can be high resolution
- population data can be high resolution
- official child poverty / deprivation statistics are typically only available as coarse regional or administrative aggregates

Because of this, existing workflows rely on relatively simple redistribution heuristics, such as using Relative Wealth Index (RWI) to spread official totals across space.

What is missing is a rigorous method that can:

- infer meaningful within-region deprivation patterns
- preserve official administrative totals
- quantify uncertainty
- show when machine learning truly adds value beyond baseline redistribution

---

## Objective

### What are we trying to achieve concretely?
We are trying to build and evaluate a research-first geospatial inference system that reconstructs **grid-level patterns of child deprivation** from coarse administrative official statistics using high-resolution proxy data.

Concretely, the project aims to:

1. combine grid-level proxies such as wealth, population, settlement type, and accessibility
2. infer relative deprivation patterns within administrative units
3. reconcile final predictions so they preserve official administrative totals exactly
4. compare this approach against simpler baselines such as uniform allocation and RWI-based redistribution
5. produce interpretable, uncertainty-aware fine-resolution maps for humanitarian prioritization

### Should sound measurable / evaluatable
The project is successful if it can be shown, using defined evaluation data and metrics, that the proposed method:

- improves fine-scale reconstruction performance relative to the RWI baseline
- preserves consistency with official statistics
- produces uncertainty-aware outputs
- remains interpretable and operationally defensible

---

## One-Line Solution

Build a **constraint-aware geospatial ML pipeline** that uses high-resolution proxy features to reconstruct grid-level child deprivation patterns from coarse official statistics, then hard-reconciles predictions back to trusted administrative totals.

---

## Proposed Approach / System Design

### Core method (ML model? pipeline? interface?)
The core method is a **geospatial machine learning pipeline** with post-processing reconciliation.

At a high level, the system works as follows:

1. assemble a base geospatial grid for Jamaica
2. attach proxy features to each grid cell:
   - Relative Wealth Index (RWI)
   - population
   - settlement class
   - travel time to cities
3. assign each grid cell to an administrative region using boundaries
4. attach administrative child poverty / deprivation targets
5. train a model to estimate relative within-region deprivation patterns
6. generate raw grid-level scores
7. apply post-processing with hard administrative reconciliation so predictions sum to official admin totals exactly
8. produce maps, uncertainty surfaces, and baseline comparisons

### Suggested model path
Start with interpretable or semi-interpretable approaches:

- regularized linear model
- GAM
- gradient boosted trees as a stronger nonlinear comparison

### Output (what the user/researcher gets)
The researcher or end user gets:

- a grid-level deprivation surface
- optionally a composite deprivation map
- comparison against uniform and RWI baselines
- uncertainty-aware outputs
- a defensible explanation of where the model adds value and where it does not

---

## Key Innovation / Why This is Novel

The novelty is not simply "using AI for poverty."

The key innovation is the combination of:

1. **coarse official statistics**
2. **high-resolution proxy data**
3. **constraint-aware reconciliation**
4. **explicit baseline comparison**
5. **uncertainty-aware humanitarian prioritization**

What makes this interesting is that the system is designed to answer a harder and more honest research question:

> When do models add information beyond redistribution?

Rather than replacing official statistics, the system acts as a bridging methodology that preserves trusted totals while learning more informative spatial allocation patterns.

This is novel because many proxy-based methods redistribute aggregate poverty without rigorously testing whether they recover meaningful multidimensional structure or simply repackage existing information.

---

## Evaluation Plan

### How will you test it?
The system will be tested through a controlled reconstruction workflow.

A practical evaluation plan is:

1. build a Jamaica geospatial feature table
2. identify the best available administrative / subregional child poverty targets
3. construct baseline methods:
   - uniform distribution
   - RWI-based redistribution
4. train one or more ML models using the same proxy features
5. produce grid-level predictions
6. reconcile all model outputs to official administrative totals
7. compare how well each method reconstructs the finer target patterns where reference data exist
8. analyze where ML improves over baseline and where it fails

### Metrics (accuracy, usability, trust, etc.)
Possible evaluation metrics include:

#### Accuracy / Reconstruction
- MAE or RMSE against finer-resolution reference targets
- correlation with known subregional deprivation values
- rank-based metrics if prioritization is the main use case

#### Baseline Improvement
- relative improvement over uniform allocation
- relative improvement over RWI redistribution

#### Constraint Compliance
- exact preservation of official administrative totals after reconciliation

#### Uncertainty Quality
- stability across runs / folds / bootstrap samples
- spread or confidence calibration where possible

#### Interpretability / Trust
- feature importance or partial dependence analysis
- qualitative defensibility for humanitarian stakeholders
- clear documentation of limitations and failure cases

---

## Current Data Available for This System

### Outcome / target data
- `Child Poverty latest estimates Feb 2026 with disaggregations.xlsx`
- `ChPov_JAM_CUB.xlsx`

These define available poverty / deprivation measures and subgroup structures.

### Proxy / feature data
- `jam_relative_wealth_index.csv`
- `jam_pop_2030_CN_100m_R2025A_v1.tif`
- `GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip`
- `cit_017_accessibility_to_cities.zip`

These define the high-resolution signals used to infer within-region variation.

### Linking / boundary data
- `gadm41_JAM.gpkg`

This makes spatial assignment and reconciliation possible.

---

## Immediate Technical Next Step

The next technical step is to build a single Jamaica feature table by:

1. inspecting all uploaded geospatial data
2. selecting the base grid
3. sampling population, settlement, and accessibility onto that grid
4. spatially assigning each observation to an admin unit
5. linking the poverty target data
6. implementing baseline redistribution methods before any ML model

This is the foundation for all later modeling and evaluation.

---

## Final Working Summary

This project aims to determine whether a constraint-aware geospatial ML system can reconstruct meaningful fine-scale child deprivation patterns from coarse official statistics better than current redistribution baselines, while preserving trusted administrative totals and producing uncertainty-aware, interpretable outputs for humanitarian prioritization.
