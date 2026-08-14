# DATA AUDIT — SwitchRank

## 1. Dataset A: Web Data Commons (WDC) Products Multi-Dimensional Benchmark

### Source & Provenance
- **Provider**: Web Data Commons (WDC) / University of Mannheim (DWS group).
- **Official Download URL**: `https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip`
- **Benchmark Paper**: Primpeli et al., *WDC Products: A Multi-Dimensional Entity Matching Benchmark*.

### Benchmark Configuration Selection
- **Variant Selected**: `80cc20rnd` (80% Hard Corner Cases, 20% Random Pairs).
- **Rationale**: Real-world e-commerce matching is dominated by subtle, hard negative pairs (near-identical titles with different specifications or model variants). Selecting `80cc20rnd` forces models to learn fine-grained attribute differences rather than relying on trivial text dissimilarity.
- **Split Sizes**:
  - `train_small`: ~2,000 candidate pairs for development.
  - `valid_small`: ~1,000 candidate pairs for validation and calibration tuning.
  - Test Sets (Official Gold Standards):
    - `000un`: 0% unseen entities in test set.
    - `050un`: 50% unseen entities in test set.
    - `100un`: 100% unseen entities in test set.

### Schema & Fields
- `id_left`, `id_right`: Unique product IDs.
- `title_left`, `title_right`: Product offer title string.
- `brand_left`, `brand_right`: Extracted brand string (may be null/missing).
- `description_left`, `description_right`: Detailed text description (may be null).
- `price_left`, `price_right`: Offer price.
- `priceCurrency_left`, `priceCurrency_right`: Currency symbol/ISO code.
- `label`: Binary target (1 = True Match, 0 = Non-Match).
- `is_hard_negative`: Boolean flag indicating hard negative pair status.

---

## 2. Dataset B: FDA AccessGUDID / 510(k) Medical Device Domain Transfer Dataset

### Source & Provenance
- **Provider**: US Food and Drug Administration (FDA) & National Library of Medicine (NLM).
- **Product Code Filter**: `FMF` ("Syringe, Piston").
- **Endpoint**: `https://api.fda.gov/device/510k.json?search=product_code:FMF&limit=1000`

### Rationale for Selection
Medical device catalog resolution is a high-impact industrial transfer domain. Product code `FMF` yields 749 canonical device identities with rich metadata (applicant, device name, catalog number, model number, clearance date).

### Perturbation & Stress Test Design
Because AccessGUDID does not provide clinical equivalence labels, we construct a deterministic canonical-resolution stress test. Each canonical device record is perturbed into heterogeneous catalog offer variants across 3 difficulty tiers:
- **EASY**: Minor whitespace, case changes, unit formatting (`10 mL` ↔ `10ml`), punctuation removal.
- **MEDIUM**: Brand abbreviation, packaging noise, trademark symbol stripping, model number separator changes.
- **HARD**: Manufacturer removed (brand only), brand removed (manufacturer only), severe descriptor shortening, numeric token reordering, combined field omission.

Ground truth identity is tied directly to the canonical `k_number` / Device Identifier.
