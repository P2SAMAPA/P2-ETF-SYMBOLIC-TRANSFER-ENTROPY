import numpy as np
import config

def discretise(series, n_bins):
    """Equal‑frequency discretisation."""
    if len(series) < n_bins:
        return np.zeros_like(series, dtype=int)
    quantiles = np.linspace(0, 100, n_bins+1)[1:-1]
    bins = np.percentile(series, quantiles)
    return np.digitize(series, bins)

def transfer_entropy(source, target, lag=1, n_bins_source=None, n_bins_target=None):
    """Transfer entropy from source to target."""
    if n_bins_source is None:
        n_bins_source = config.ETF_BINS
    if n_bins_target is None:
        n_bins_target = config.ETF_BINS
    # Discretise
    src = discretise(source, n_bins_source)
    tgt = discretise(target, n_bins_target)
    if len(src) <= lag:
        return 0.0
    # Align
    src_past = src[:-lag]
    tgt_past = tgt[:-lag]
    tgt_future = tgt[lag:]
    # Compute joint counts for conditional entropy H(tgt_future | tgt_past)
    counts_yy = {}
    for yf, yp in zip(tgt_future, tgt_past):
        counts_yy[(yf, yp)] = counts_yy.get((yf, yp), 0) + 1
    total = len(tgt_future)
    H_cond1 = 0.0
    for (yf, yp), cnt in counts_yy.items():
        pyf_yp = cnt / sum(1 for (_, yp_) in counts_yy if yp_ == yp)
        pyf_yp = max(pyf_yp, 1e-12)
        H_cond1 += (cnt / total) * (-np.log2(pyf_yp))
    # Conditional entropy H(tgt_future | src_past, tgt_past)
    counts_xyy = {}
    for yf, xp, yp in zip(tgt_future, src_past, tgt_past):
        counts_xyy[(yf, xp, yp)] = counts_xyy.get((yf, xp, yp), 0) + 1
    H_cond2 = 0.0
    for (yf, xp, yp), cnt in counts_xyy.items():
        denom = sum(1 for (_, xp_, yp_) in counts_xyy if xp_ == xp and yp_ == yp)
        p_yf_xp_yp = cnt / denom if denom > 0 else 0
        p_yf_xp_yp = max(p_yf_xp_yp, 1e-12)
        H_cond2 += (cnt / total) * (-np.log2(p_yf_xp_yp))
    te = max(H_cond1 - H_cond2, 0.0)
    return te

def conditional_transfer_entropy(source, target, macro_symbols, macro_bins=3, lag=1):
    """
    Compute TE from source (macro) to target (ETF) conditioned on the macro symbol at time t.
    Returns: average TE over macro symbols, and also TE for each symbol.
    """
    if len(source) != len(target):
        min_len = min(len(source), len(target))
        source = source[:min_len]
        target = target[:min_len]
        macro_symbols = macro_symbols[:min_len]
    if len(source) < lag + 2:
        return 0.0, {}
    # Discretise source and target
    src_disc = discretise(source, config.ETF_BINS)
    tgt_disc = discretise(target, config.ETF_BINS)
    # Separate by macro symbol
    te_by_symbol = {}
    for sym in range(macro_bins):
        idx = (macro_symbols == sym)
        if np.sum(idx) < lag + 2:
            te = 0.0
        else:
            src_sub = src_disc[idx]
            tgt_sub = tgt_disc[idx]
            te = transfer_entropy(src_sub, tgt_sub, lag=lag, n_bins_source=config.ETF_BINS, n_bins_target=config.ETF_BINS)
        te_by_symbol[sym] = te
    # Average TE across symbols (weighted by frequency)
    weights = np.bincount(macro_symbols, minlength=macro_bins) / len(macro_symbols)
    avg_te = sum(te_by_symbol[sym] * weights[sym] for sym in range(macro_bins))
    return avg_te, te_by_symbol

def symbolic_te_score(returns, macro_series, use_conditional_on_today=True, macro_bins=3, lag=1):
    """
    Compute score for one ETF:
    - If use_conditional_on_today: return TE for the macro symbol corresponding to today's macro value.
    - Else: return average TE across macro symbols.
    """
    if len(returns) < lag + 5 or len(macro_series) < lag + 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_series))
    returns = returns[:min_len]
    macro_series = macro_series[:min_len]
    # Discretise macro into symbols
    macro_symbols = discretise(macro_series, macro_bins)
    # Compute conditional TE
    avg_te, te_by_sym = conditional_transfer_entropy(macro_series, returns, macro_symbols, macro_bins, lag)
    if use_conditional_on_today:
        today_macro = macro_series[-1]
        # Find which bin today's macro belongs to
        # We need the same discretisation as used above; we recompute bins from the series
        quantiles = np.linspace(0, 100, macro_bins+1)[1:-1]
        bins = np.percentile(macro_series, quantiles)
        today_sym = np.digitize(today_macro, bins)
        # Ensure within 0..macro_bins-1
        today_sym = min(max(today_sym, 0), macro_bins-1)
        score = te_by_sym.get(today_sym, 0.0)
    else:
        score = avg_te
    return float(score)
