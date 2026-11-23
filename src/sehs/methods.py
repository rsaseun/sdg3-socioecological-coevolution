"""Statistical methods used in the study (one place, unit-testable).

Established methods: nonparanormal graphical lasso with cross-validated selection, bridge centrality, case-dropping stability,
entropy-weighted coupling coordination degree, beta/sigma convergence, discrete Markov chains, two-way fixed
effects with Driscoll-Kraay errors, Dumitrescu-Hurlin panel Granger non-causality test, obstacle degree model.
Newer combinations: node predictability and module recovery (adjusted Rand index) of theoretical socioecological
levels; regularised first-difference panel VAR (lead-lag network) with country-block bootstrap; spatial Markov
chains of coordination states; modified coupling degree; within-between (Mundlak) decomposition of the
coordination dividend.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import graphical_lasso


# =============================================================================================== networks
def normal_scores(X):
    """Nonparanormal transform: rank-based inverse normal scores, column-standardised."""
    X = np.asarray(X, float)
    r = np.apply_along_axis(stats.rankdata, 0, X)
    Z = stats.norm.ppf((r - 0.5) / X.shape[0])
    return (Z - Z.mean(0)) / Z.std(0)


def _pcor_from_prec(prec, thr=0.0):
    d = np.sqrt(np.diag(prec))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 0)
    pc[np.abs(pc) < thr] = 0
    return pc


def cv1se_penalty(Z, seed, n_splits=5, n_repeats=10, n_alpha=40):
    """Graphical-lasso penalty by repeated k-fold cross-validated likelihood with the one-standard-error rule.

    The 1-SE rule picks the sparsest model whose held-out likelihood is within one standard error of the best,
    a conservative choice suited to small samples (here 48 countries).
    """
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.model_selection import RepeatedKFold
    S = np.corrcoef(Z, rowvar=False)
    lam_max = np.max(np.abs(S - np.eye(len(S))))
    alphas = np.exp(np.linspace(np.log(0.7 * lam_max), np.log(0.02), n_alpha))
    m = GraphicalLassoCV(alphas=alphas, cv=RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                                         random_state=seed), max_iter=500).fit(Z)
    r = m.cv_results_
    mu, se = r["mean_test_score"], r["std_test_score"] / np.sqrt(n_splits * n_repeats)
    best = int(np.argmax(mu))
    ok = np.where(mu >= mu[best] - se[best])[0]
    return float(r["alphas"][ok].max()), float(r["alphas"][best])


def glasso_net(Z, alpha, thr=0.05):
    """Graphical lasso at a fixed penalty; partial correlations below |thr| are treated as absent.

    Returns partial correlations, precision matrix and node predictability (R^2 of each node from all others).
    """
    S = np.corrcoef(Z, rowvar=False)
    cov, prec = graphical_lasso(S, alpha=alpha, max_iter=500, tol=1e-4)
    pc = _pcor_from_prec(prec, thr)
    r2 = 1 - 1 / np.diag(prec)
    return pc, prec, r2


def bridge_strength(W, groups):
    """Sum of absolute edge weights from each node to nodes of other groups (Jones et al., 2021)."""
    g = np.asarray(groups)
    return np.array([np.abs(W[i, g != g[i]]).sum() for i in range(len(g))])


def louvain_modules(W, seed, resolution=1.0):
    import networkx as nx
    G = nx.Graph()
    p = W.shape[0]
    G.add_nodes_from(range(p))
    for i, j in itertools.combinations(range(p), 2):
        if W[i, j] > 0:  # positive associations define cohesive modules
            G.add_edge(i, j, weight=W[i, j])
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    lab = np.zeros(p, int)
    for k, c in enumerate(sorted(comms, key=lambda c: -len(c))):
        lab[list(c)] = k
    Q = nx.community.modularity(G, comms, weight="weight")
    return lab, Q, G


def modularity_of(G, labels):
    import networkx as nx
    parts = [set(np.where(labels == k)[0]) for k in np.unique(labels)]
    return nx.community.modularity(G, parts, weight="weight")


# =============================================================================================== coordination
def minmax_pooled(df, direction, lo_q=0.01, hi_q=0.99):
    """Pooled (all country-years) goalposts so that index values are comparable over time."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        lo, hi = df[c].quantile(lo_q), df[c].quantile(hi_q)
        x = (df[c].clip(lo, hi) - lo) / (hi - lo)
        out[c] = x if direction[c] > 0 else 1 - x
    return 0.01 + 0.99 * out  # strictly positive for entropy and geometric means


def entropy_weights(Xn):
    P = Xn / Xn.sum(0)
    k = 1 / np.log(len(Xn))
    e = -k * (P * np.log(P)).sum(0)
    d = 1 - e
    return d / d.sum()


def critic_weights(Xn):
    """CRITIC weights (Diakoulaki et al., 1995): contrast intensity x conflict with other criteria."""
    sd = Xn.std(0)
    R = np.corrcoef(Xn, rowvar=False)
    c = sd * (1 - R).sum(0)
    return c / c.sum()


def pca_weights(Xn):
    Z = (Xn - Xn.mean(0)) / Xn.std(0)
    ev, vec = np.linalg.eigh(np.cov(Z, rowvar=False))
    v = np.abs(vec[:, -1])
    return v / v.sum()


def ccd(U):
    """Coupling (C), comprehensive development (T) and coupling coordination degree (D) for n subsystems."""
    U = np.asarray(U, float)
    n = U.shape[1]
    C = np.exp(np.log(U).mean(1)) / U.mean(1)
    T = U.mean(1)
    return C, T, np.sqrt(C * T)


def ccd_modified(U):
    """Modified coupling degree correcting the compressed range of the classical C (Wang et al., 2021)."""
    U = np.asarray(U, float)
    n = U.shape[1]
    pair = sum(np.abs(U[:, i] - U[:, j]) for i, j in itertools.combinations(range(n), 2))
    term1 = 1 - pair / sum(range(1, n))
    term2 = np.prod(U / U.max(1, keepdims=True), axis=1) ** (1 / (n - 1))
    C = np.sqrt(np.clip(term1 * term2, 0, 1))
    T = U.mean(1)
    return C, T, np.sqrt(C * T)


def classify(D, classes):
    lab = np.empty(len(D), object)
    for lo, hi, name in classes:
        lab[(D >= lo) & (D < hi)] = name
    return lab


def obstacle_degree(Xn_row, w):
    """Obstacle degree of each indicator: weighted shortfall share (Yang et al. style obstacle model)."""
    dev = w * (1 - Xn_row)
    return dev / dev.sum()


# =============================================================================================== panel time series
def dumitrescu_hurlin(df, y, x, unit="iso3", time="year", K=1):
    """Dumitrescu & Hurlin (2012) test of H0: x does not Granger-cause y for any unit.

    Heterogeneous unit-wise OLS of y_t on K lags of y and x; W-bar is the mean unit Wald statistic.
    Returns W-bar, Z-bar, Z-bar-tilde (small-T corrected) and p-values.
    """
    W, Ts = [], []
    for _, g in df.sort_values(time).groupby(unit):
        g = g[[y, x]].dropna()
        if len(g) < 3 * K + 5:
            continue
        Y = g[y].values
        lag = lambda v, k: v[K - k:len(v) - k]
        X = np.column_stack([np.ones(len(Y) - K)] + [lag(g[y].values, k) for k in range(1, K + 1)] +
                            [lag(g[x].values, k) for k in range(1, K + 1)])
        yy = Y[K:]
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        e = yy - X @ b
        dof = len(yy) - X.shape[1]
        s2 = e @ e / dof
        try:
            V = s2 * np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        R = np.zeros((K, X.shape[1]))
        for k in range(K):
            R[k, 1 + K + k] = 1
        rb = R @ b
        W.append(float(rb @ np.linalg.inv(R @ V @ R.T) @ rb))
        Ts.append(len(yy))
    N, T = len(W), int(np.median(Ts))
    Wbar = float(np.mean(W))
    Zbar = np.sqrt(N / (2 * K)) * (Wbar - K)
    Ztil = np.sqrt(N / (2 * K)) * (T - 2 * K - 5) / (T - K - 3) * ((T - 2 * K - 3) / (T - 2 * K - 1) * Wbar - K)
    return dict(N=N, T=T, K=K, Wbar=Wbar, Zbar=float(Zbar), p_Zbar=float(2 * (1 - stats.norm.cdf(abs(Zbar)))),
                Ztilde=float(Ztil), p_Ztilde=float(2 * (1 - stats.norm.cdf(abs(Ztil)))))


def ips_like(df, var, unit="iso3", time="year"):
    """Mean of unit-wise ADF t-statistics (Im-Pesaran-Shin style summary) with the share of units rejecting."""
    from statsmodels.tsa.stattools import adfuller
    ts, rej = [], []
    for _, g in df.sort_values(time).groupby(unit):
        v = g[var].dropna().values
        if np.std(v) < 1e-9:
            continue
        r = adfuller(v, maxlag=1, regression="c", autolag=None)
        ts.append(r[0]); rej.append(r[1] < 0.05)
    return float(np.mean(ts)), float(np.mean(rej))
