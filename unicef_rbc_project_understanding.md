# UNICEF × RBC Borealis AI Project
## Current Understanding, Data Inventory, and Technical Next Steps

---

## 1. Project Understanding So Far

### Project Theme
This project sits at the intersection of:

- humanitarian forecasting
- child vulnerability / deprivation estimation
- geospatial machine learning
- constrained statistical inference under data scarcity

It is framed as a **research-first system design project** with real operational relevance for UNICEF-style disaster preparedness and resource prioritization.

---

## 2. What We Know About the Problem

### Core Problem
UNICEF and similar humanitarian systems need to identify **where vulnerable children are located at fine spatial resolution**, especially before or during climate-related disasters.

However, the key inputs exist at very different resolutions:

- **Hazard / climate exposure data** can be high resolution
- **Population data** can be high resolution
- **Official child poverty / deprivation data** are usually only available at coarse administrative levels

This creates a major **resolution mismatch**.

### Why This Matters
In practice, decision-makers may want to know:

- which communities along a likely hurricane path are most vulnerable
- where pre-positioned supplies should go
- which locations contain the highest concentrations of poor or deprived children
- which schools or service catchments may be most affected

Today, coarse official poverty statistics are not spatially precise enough to answer these questions well.

---

## 3. What Is Missing Today

### Current Limitation
The current workaround is effectively a redistribution exercise:

- use official aggregate poverty totals
- use proxy data like Relative Wealth Index (RWI)
- spread those totals across space in a more informed way

But this raises a scientific and operational question:

> Are we learning meaningful fine-scale deprivation patterns, or simply reshaping existing aggregates?

### The Gap
What is missing today is a rigorous, constraint-aware method that can:

1. use high-resolution proxies to infer **within-region variation**
2. preserve official administrative totals
3. support grid-level prioritization
4. quantify uncertainty
5. show when machine learning adds value beyond simple redistribution baselines

---

## 4. What We Are Trying to Achieve

### Concrete Objective
Build and evaluate a system that uses high-resolution geospatial proxy data to reconstruct **grid-level patterns of multidimensional child deprivation** from coarse administrative official statistics.

### Research Orientation
The primary goal is **research**, not deployment-first productionization.

The main evaluation question is:

> Can machine learning produce more useful fine-scale deprivation estimates than the current RWI redistribution baseline?

### Output Type
The current intended output is:

- **grid-level predictions**
- interpretable enough for humanitarian use
- uncertainty-aware
- administratively reconciled to official totals

---

## 5. Locked Design Decisions So Far

### Output
- Grid-level predictions

### Primary Orientation
- Research-first

### Operational Use
- Precomputed vulnerability / deprivation surfaces

### Constraint Strategy
- **Post-processing with hard administrative reconciliation**

This means:

1. the model first predicts relative fine-scale variation
2. predictions are then rescaled within each administrative region
3. final predictions exactly preserve trusted administrative totals

### Interpretability
- Required

### Success Criteria
- Beat RWI redistribution baseline
- Produce uncertainty-aware maps
- Tell a strong, honest humanitarian impact story
- Show where ML works and where it does not

### Positioning
The system is **not replacing official statistics**.
It is a bridging methodology for prioritization under uncertainty.

---

## 6. Recommended Target Strategy

Although a composite multidimensional output is important operationally, the stronger research strategy is:

### Model dimensions separately where possible
- WASH
- health
- education
- or, based on available data, severe prevalence / moderate prevalence / depth measures separately

### Then derive a composite output afterward

Why this is better:

- more interpretable
- easier to diagnose failure modes
- more honest scientifically
- prevents hiding weak model behavior inside a single composite index

---

## 7. Data We Know About So Far

The project currently has two major categories of data:

### A. Outcome / Target Data
Administrative or subgroup-level child poverty / deprivation tables

### B. Spatial Proxy Data
Grid-level geospatial features used to infer fine-scale variation

### C. Linking / Boundary Data
Spatial boundaries needed to connect grid cells to official administrative regions

---

## 8. Uploaded Data Inventory

---

### 8.1 `UNICEF FDN - RBC LSI Spring 26.docx`
### What it is
A technical concept / proposal document describing the scientific and methodological framing of the project.

### What it told us
- the project is about reconstructing multidimensional child deprivation under data scarcity
- the deprivation dimensions of interest include WASH, health, and education
- the main issue is the mismatch between fine-resolution hazard / population data and coarse administrative deprivation data
- baseline approaches include uniform allocation and RWI-based redistribution
- candidate models include regularized regression, GAMs, boosted trees, and neural networks
- evaluation should include per-dimension accuracy, structural fidelity, and improvement over baselines

### Why it is important
This file defines the **research question**, the **scientific framing**, and the **evaluation logic**.

---

### 8.2 `UNICEF FDN - RBC Borealis LSI Spring 2026 - Kick off.pptx`
### What it is
A kickoff deck giving the operational context and problem motivation.

### What it told us
- climate emergencies disproportionately affect children
- current systems estimate people in need / children in need using poverty as a proxy for coping capacity
- official poverty data are only available at coarse administrative levels
- UNICEF currently uses RWI-based spatial redistribution as a workaround
- the challenge is to move from regional poverty totals to local prioritization

### Why it is important
This file grounds the project in a **real humanitarian workflow** and clarifies the **use case** for spatial prioritization.

---

### 8.3 `Child Poverty latest estimates Feb 2026 with disaggregations.xlsx`
### What it is
A large cross-country administrative / survey-based child poverty table.

### What we know about it
- contains broad international coverage
- includes multiple survey years
- includes survey sources such as DHS and MICS
- includes subgroup disaggregations such as sex, residence, wealth quintile, and combinations
- includes measures such as severe prevalence, moderate prevalence, depth, and distributions across deprivation counts

### Why it is important
This is the broad **target-side reference table** that supports:
- outcome definition
- subgroup logic
- country filtering
- potential training / validation design across countries

It tells us what deprivation outputs are available and how they are structured.

---

### 8.4 `ChPov_JAM_CUB.xlsx`
### What it is
A more spatially useful child poverty file focused on Jamaica and Cuba.

### What we know about it
- includes country codes for JAM and CUB
- includes subregional information
- contains poverty-related outputs at finer subnational resolution than the broad global table

### Why it is important
This file is especially valuable for:
- controlled experiments
- spatial coarsening / reconstruction exercises
- early validation and debugging
- a feasible first-country workflow

It is a bridge between national-level statistics and finer subregional evaluation.

---

### 8.5 `jam_relative_wealth_index.csv`
### What it is
A Jamaica-specific Relative Wealth Index file.

### What it likely provides
- grid-level or point-level wealth proxy values
- coordinate-linked observations that can serve as the project's base geospatial grid

### Why it is important
This is one of the most important files in the entire system because it plays two roles:

1. **Core model feature**
2. **Main baseline comparator**

If the ML model cannot outperform an RWI-based redistribution baseline, then the added complexity may not be justified.

This file is likely the best candidate for the **base grid** onto which other geospatial layers will be aligned.

---

### 8.6 `jam_pop_2030_CN_100m_R2025A_v1.tif`
### What it is
A high-resolution Jamaica population raster from WorldPop or a similar source.

### What it provides
- population counts at fine spatial resolution
- potentially child-focused population information, depending on the product

### Why it is important
Population is necessary for:
- realistic weighting
- aggregation
- interpreting deprivation in terms of affected people, not just grid scores
- reconciliation and prioritization

Without population, a vulnerability surface can become spatially interesting but operationally weak.

---

### 8.7 `GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip`
### What it is
A GHSL SMOD settlement classification dataset.

### What it provides
Settlement / urbanization structure, such as:
- urban center
- urban cluster
- rural classes

### Why it is important
Settlement structure is a strong predictor of deprivation patterns and service access. It helps the model distinguish between:
- dense urban settings
- peri-urban environments
- rural or remote areas

This is especially relevant because urban/rural distinctions already appear in the poverty data logic and current UNICEF workflows.

---

### 8.8 `gadm41_JAM.gpkg`
### What it is
A Jamaica administrative boundary file.

### What it provides
- official geographic boundaries
- multiple admin levels, depending on the package contents

### Why it is important
This file is essential for the system to function technically.

It is the spatial linking layer that allows us to:
- assign each geospatial observation / grid cell to an administrative unit
- merge grid-level proxies with admin-level poverty targets
- perform hard administrative reconciliation

Without boundaries, the system cannot connect fine-scale predictions to official statistics.

---

### 8.9 `cit_017_accessibility_to_cities.zip`
### What it is
A travel-time-to-cities accessibility dataset.

### What it provides
- estimated travel time to the nearest city or urban center

### Why it is important
This adds a critical signal that RWI alone does not fully capture:
- remoteness
- connectivity
- structural access to services and economic centers

This feature is one of the best candidates for helping the model beat an RWI-only baseline, because it captures **access constraints**, not just wealth status.

---

## 9. What the Data Stack Means Systemically

We now have a viable minimum stack for Jamaica:

### Outcome / target side
- child poverty / deprivation tables

### Grid-level proxy side
- RWI
- population
- settlement structure
- accessibility

### Linking side
- admin boundaries

This means the project now has enough pieces to build a real **geospatial ML pipeline**.

What is still hard is not finding data, but:
- aligning the datasets correctly
- defining the training target properly
- evaluating whether the model adds value beyond redistribution

---

## 10. Technical Interpretation of the Learning Task

The model is **not** directly estimating official truth from scratch.

Instead, it is learning:

> the relative spatial allocation of deprivation within administrative units

Then, after prediction, outputs are constrained so they match trusted official totals.

This is a much more defensible scientific and operational framing.

---

## 11. What Needs to Be Done Next, Technically

### Step 1 — Inspect every file
- confirm formats
- inspect columns, CRS, resolution, extent, missing values
- verify what the RWI CSV actually contains
- inspect what administrative levels exist in the GADM geopackage
- inspect raster metadata for population, SMOD, and accessibility layers

### Step 2 — Choose a base spatial representation
Recommended:
- use the RWI file as the base grid if it contains coordinates / cell-level observations

Why:
- it is already central to both the baseline and model
- other layers can be sampled onto it

### Step 3 — Standardize coordinate systems
- ensure all geospatial layers are in compatible CRS
- reproject boundaries and rasters as needed

### Step 4 — Build the Jamaica feature table
For each base-grid observation, attach:
- latitude / longitude or geometry
- RWI value
- sampled population value
- sampled settlement class
- sampled travel-time value

### Step 5 — Spatially assign admin regions
Using `gadm41_JAM.gpkg`:
- assign every grid observation to its admin unit
- determine the admin level that matches the poverty targets best

### Step 6 — Prepare target tables
From the poverty spreadsheets:
- isolate Jamaica rows
- identify matching admin / subregion units
- choose the initial target variables
- reconcile naming / coding differences between spreadsheet regions and boundary file regions

### Step 7 — Join targets to the feature table
Each grid observation should inherit the target associated with its admin region.

### Step 8 — Define baseline methods
At minimum:
- uniform allocation baseline
- RWI-based redistribution baseline

These are necessary before any ML modeling.

### Step 9 — Define the first ML model
Start simple and interpretable:
- ridge / regularized linear model
- GAM
- gradient boosted trees if needed after that

Use features such as:
- RWI
- population
- settlement
- travel time

### Step 10 — Implement post-processing reconciliation
After producing raw grid-level scores:
- rescale predictions within each admin region
- force them to sum to the official admin total exactly

This is a required part of the final system design.

### Step 11 — Define evaluation setup
Potential evaluation structure:
- use subregional Jamaican data where finer truth is available
- compare:
  - uniform baseline
  - RWI baseline
  - ML model
- assess whether ML improves fine-scale reconstruction

### Step 12 — Add uncertainty estimation
For the MVP, uncertainty can come from:
- model ensembles
- bootstrap resampling
- variability across folds / seeds

### Step 13 — Produce outputs
Final outputs should include:
- grid-level deprivation map
- optionally composite deprivation map
- uncertainty map
- baseline comparison results

---

## 12. Practical Priority Order

If time is limited, do things in this order:

1. inspect data
2. build one clean Jamaica feature table
3. assign admin regions
4. build uniform and RWI baselines
5. define one initial target
6. train one interpretable model
7. reconcile to admin totals
8. evaluate against baselines
9. add uncertainty
10. improve / expand only if justified

---

## 13. Major Risks to Watch Closely

### Risk 1 — You might just recreate RWI
If the model adds no value beyond RWI, that is still a valid research result.

### Risk 2 — Region matching may be messy
Administrative names / codes across datasets may not line up cleanly.

### Risk 3 — Feature-resolution mismatch
Different rasters may not align naturally.

### Risk 4 — False precision
A fine-resolution map may look more certain than it actually is.

### Risk 5 — Weak evaluation design
If the validation setup is weak, claims about model improvement will not hold.

---

## 14. Current Best Framing of the Project

A strong current framing is:

> This project builds a constraint-aware geospatial machine learning system that uses high-resolution proxy data to reconstruct fine-scale spatial patterns of child deprivation from coarse official statistics, while preserving trusted administrative totals and evaluating whether ML adds value beyond RWI-based redistribution.

