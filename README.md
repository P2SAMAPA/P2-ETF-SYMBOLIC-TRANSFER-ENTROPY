# Symbolic Transfer Entropy with Macro State

Extends transfer entropy by conditioning on the macro state itself. The macro variable (e.g., VIX) is discretised into symbols (low, medium, high). For each symbol, we compute TE(macro → ETF) using only days when macro was in that state. The per‑ETF score is the TE conditional on today's macro symbol – a non‑linear, state‑dependent predictability signal.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Equal‑frequency binning for macro and ETF returns
- Transfer entropy with configurable lag
- Score = conditional TE on today's macro symbol (or average over symbols)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-symbolic-transfer-entropy-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High conditional TE → macro strongly predicts ETF returns given the current macro level (e.g., high VIX).
- Low TE → macro is not predictive in that regime.
- This engine reveals non‑linear relationships missed by standard transfer entropy.

## Requirements

See `requirements.txt`.
