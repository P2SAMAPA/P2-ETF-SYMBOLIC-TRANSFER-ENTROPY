import numpy as np
import config   # <-- added import

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
    src = discretise(source, n_bins_source)
    tgt = discretise(target, n_bins_target)
    if len(src) <= lag:
        return 0.0
    src_past = src[:-lag]
    tgt_past = tgt[:-lag]
    tgt_future = tgt[lag:]
    # H(tgt_future | tgt_past)
    counts_yy = {}
    for yf, yp in zip(tgt_future, tgt_past):
        counts_yy[(yf, yp)] = counts_yy.get((yf, yp), 0) + 1
    total = len(tgt_future)
    H_cond1 = 0.0
    for (yf, yp), cnt in counts_yy.items():
        pyf_yp = cnt / sum(1 for (_, yp_) in counts_yy if yp_ == yp)
        pyf_yp = max(pyf_yp, 1e-12)
        H_cond1 += (cnt / total) * (-np.log2(pyf_yp))
    # H(tgt_future | src_past, tgt_past)
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
    src_disc = discretise(source, config.ETF_BINS)  # now config is imported
    tgt_disc = discretise(target, config.ETF_BINS)
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
    weights = np.bincount(macro_symbols, minlength=macro_bins) / len(macro_symbols)
    avg_te = sum(te_by_symbol[sym] * weights[sym] for sym in range(macro_bins))
    return avg_te, te_by_symbol

def symbolic_te_score(returns, macro_df, use_conditional_on_today=True, macro_bins=3, lag=1):
    """
    Compute score for one ETF by averaging conditional transfer entropy over all macro variables.
    """
    if len(returns) < lag + 5 or macro_df is None or len(macro_df) < lag + 5:
        return 0.0
    te_values = []
    for macro_col in macro_df.columns:
        macro_series = macro_df[macro_col].values
        if len(macro_series) != len(returns):
            min_len = min(len(returns), len(macro_series))
            macro_series = macro_series[:min_len]
            rets = returns[:min_len]
        else:
            rets = returns
        if len(rets) < lag + 5:
            continue
        macro_symbols = discretise(macro_series, macro_bins)
        avg_te, te_by_sym = conditional_transfer_entropy(macro_series, rets, macro_symbols, macro_bins, lag)
        if use_conditional_on_today:
            today_macro = macro_series[-1]
            quantiles = np.linspace(0, 100, macro_bins+1)[1:-1]
            bins = np.percentile(macro_series, quantiles)
            today_sym = np.digitize(today_macro, bins)
            today_sym = min(max(today_sym, 0), macro_bins-1)
            te = te_by_sym.get(today_sym, 0.0)
        else:
            te = avg_te
        te_values.append(te)
    if not te_values:
        return 0.0
    return float(np.mean(te_values))
