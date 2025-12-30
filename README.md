# Attention Lifespan: Modeling Public Attention Dynamics from Search Data

## Overview

This project studies how public attention to discrete news events evolves over time. Using daily Google search interest data for major U.S. political events, I construct quantitative measures of **attention magnitude**, **persistence**, and **decay**, and analyze how these components interact to explain total attention received by an event.

The goal is not prediction, but **measurement and structure**: to understand *what governs the lifespan of attention* and whether attention behaves differently across distinct regimes (e.g., short-lived shocks versus persistent narratives).

This project is motivated by broader questions in statistical learning and time-series modeling:

- How should transient vs. persistent signals be distinguished?
- What summary statistics preserve meaningful structure in noisy temporal data?
- When does model simplicity outperform more expressive formulations?

---

## Data

The dataset consists of daily Google search interest (indexed to 100) for **40 major U.S. political events** between January 1, 2017 and September 1, 2017.

Each column represents a distinct event (e.g., *Charlottesville*, *Women's March*, *James Comey fired*), and each row represents daily search interest.

This type of data is:

- Sparse
- Heavy-tailed
- Dominated by short-lived spikes

which makes it well-suited for studying attention decay dynamics.

---

## Metrics Constructed

For each event, I compute the following interpretable metrics:

**Peak Attention**  
Maximum observed search interest.

**Half-Life (days)**  
Number of days after the peak until attention falls below 50% of peak.

**Total Attention (AUC)**  
Area under the attention curve, measuring cumulative exposure.

**Exponential Decay Rate (λ)**  
Estimated decay parameter assuming post-peak exponential decline.

These metrics compress high-dimensional time series into quantities that retain temporal meaning while enabling cross-event comparison.

---

## Modeling Approach

### 1. Linear Explanations of Total Attention

I first examine whether total attention (AUC) is better explained by peak magnitude or attention persistence (half-life).

**Result:**

- AUC is strongly explained by peak magnitude (R² ≈ 0.78)
- Half-life alone explains little variance (R² ≈ 0.03)

This suggests that *how loud* an event is initially matters more than *how long* it lingers—at least globally.

### 2. Multiplicative Structure

I then test a simple multiplicative model:

```
AUC ≈ Peak × (1/λ)
```

which corresponds to an exponential decay assumption.

**Result:**

- This model performs poorly overall (R² ≈ 0.17)
- Indicates that attention dynamics are not governed by a single decay mechanism

This failure motivates regime separation.

### 3. Regime Identification: Shock vs. Persistent Events

Events are classified into two regimes based on decay rate:

- **Shock regime**: fast decay (high λ)
- **Persistent regime**: slow decay (low λ)

This separation reveals **structural heterogeneity** in attention dynamics.

---

## Key Findings

### Regime-Level Results

**Shock Regime**

- Short-lived, fast-decaying spikes
- AUC strongly explained by peak alone (R² ≈ 0.87)

**Persistent Regime**

- Lower decay, sustained interest
- AUC still explained by peak, but with different slope (R² ≈ 0.76)

This shows that:

- A single global model obscures structure
- Simple models perform well *once the data-generating regime is identified*

---

## Why This Matters

This project demonstrates a core principle in statistical learning:

> **Correct structure often matters more than model complexity.**

Rather than applying increasingly flexible models, identifying latent regimes yields:

- Better interpretability
- Stronger explanatory power
- Clearer causal narratives

The same principle applies broadly to time-series modeling, anomaly detection, regime-switching systems, and learning under nonstationarity.

---

## What This Project Demonstrates

From an admissions perspective, this project reflects my approach to research:

- I start from a **clear, well-posed question**
- Construct **interpretable mathematical summaries**
- Test simple models before escalating complexity
- Use failures to motivate better structure
- Treat data analysis as hypothesis-driven, not exploratory noise

The emphasis is on **understanding**, not performance chasing.

---

## Future Directions

Possible extensions include:

- Hidden Markov or changepoint models for regime detection
- Hierarchical modeling across event types
- Comparing political vs. non-political attention dynamics
- Cross-platform attention (search vs. social media)

---

## Reproducibility

All results can be reproduced by running:

```bash
python compute_metrics.py
```

The script loads the raw data, computes all metrics, performs regressions, and prints summarized results.
