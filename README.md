# Hybrid Logic-Constrained Explainer

This project implements a **Hybrid Explainability Framework** that combines Formal Logic (XReason) with Heuristic Explainers (LIME, SHAP, Anchors). 

By constraining the sampling space of heuristic explainers to a formally verified "Safe Zone" (Hyperbox), this tool eliminates "hallucinations" (out-of-distribution errors) and guarantees **0% counterexamples**, offering a solution that is faster than pure formal minimization and more reliable than standard heuristics.

---

## 1. Build and Installation

### Prerequisites
* Python 3.8+
* **XReason:** An external formal logic tool for extracting prime implicants from tree ensembles. 
    * *Note:* Ensure the path to the XReason runner is correctly set in `src/xreason_interface.py` (Default: `/home/vivresavie/xreason/src/xreason.py`).

### Standard Build Instructions
We use `pip` for dependency management.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd explainers-project

# 2. Install Python Dependencies
pip install -r requirements.txt