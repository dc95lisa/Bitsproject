# Bitsproject

## Custom Neural Network Use Case: Grocery Chain Demand Forecasting

A realistic organization-level use case is a **national grocery retailer** forecasting SKU-level demand per store for the next day/week.

### Why a custom neural network?
Classical models (ARIMA/Prophet/XGBoost alone) can struggle to jointly model:
- Store attributes (size, region, demographics)
- Product attributes (category, perishability, promotion sensitivity)
- Temporal signals (recent demand, day-of-week, seasonality)
- Nonlinear interactions (promotion effect differs by store/product)

A custom neural network can combine these inputs through dedicated feature towers and a shared prediction head.

---

## GPU Requirements During Training

When estimating GPU requirements, plan for these memory components:

1. **Model parameters** (weights)
2. **Gradients** (roughly similar size as weights)
3. **Optimizer states** (Adam/AdamW often ~2x parameter memory)
4. **Activations** (depends heavily on batch size and network depth)
5. **Framework overhead + fragmentation** (safety margin)

A practical rough formula used in this repo script:

- `Total VRAM ≈ (params + grads + optimizer + activations) * safety_factor`

### Quick sizing heuristics
- Small tabular model (few million params): 8–16 GB GPU usually enough.
- Medium multimodal model (10M–100M params): 24–48 GB often needed.
- Large models: multi-GPU with data/model parallelism.

Use mixed precision (FP16/BF16), gradient checkpointing, and gradient accumulation if memory is tight.

---

## Main Bottlenecks (Real Life)

1. **Data quality & feature freshness**
   - Inconsistent POS data, stockouts mislabeled as low demand, delayed promotion feeds.
2. **Training throughput**
   - Slow dataloader or feature joins can leave GPU idle.
3. **Concept drift**
   - Demand behavior changes (holidays, inflation, local events).
4. **Latency at inference**
   - Store planning systems may require predictions within strict SLA windows.
5. **MLOps governance**
   - Versioning, rollback, explainability, and audit trails for business trust.

---

## How to Take It to Production

1. **Data & feature pipeline**
   - Build reproducible offline/online feature definitions.
   - Add validation checks (nulls, ranges, drift).
2. **Training pipeline**
   - Containerized training job with experiment tracking.
   - Save metrics and artifacts in model registry.
3. **Validation gate**
   - Promote only if model beats baseline (MAPE/WAPE/stockout metrics).
4. **Serving pattern**
   - Batch inference nightly for replenishment planning.
   - Optional realtime endpoint for dynamic repricing decisions.
5. **Progressive rollout**
   - Shadow mode → limited region rollout → full rollout.
6. **Monitoring**
   - Data drift, prediction drift, business KPIs, SLA latency.
7. **Retraining strategy**
   - Scheduled retraining (weekly) + event-triggered retraining.

---

## Code

See:

- `custom_nn_gpu_prod_use_case.py`

The script includes:
- A GPU memory estimation helper
- A custom PyTorch demand network
- Mixed-precision training loop
- Production handoff placeholder for export/serving

Run:

```bash
python custom_nn_gpu_prod_use_case.py
```
