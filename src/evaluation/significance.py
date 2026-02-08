import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def paired_t_test(baseline: List[float], ore: List[float]) -> Tuple[float, float]:
    t_stat, p_value = stats.ttest_rel(baseline, ore)
    return t_stat, p_value


def wilcoxon_test(baseline: List[float], ore: List[float]) -> Tuple[float, float]:
    diff = np.array(ore) - np.array(baseline)
    non_zero = diff != 0
    
    if sum(non_zero) < 10:
        return np.nan, np.nan
    
    stat, p_value = stats.wilcoxon(
        np.array(baseline)[non_zero], 
        np.array(ore)[non_zero]
    )
    return stat, p_value


def bootstrap_ci(baseline: List[float], ore: List[float], 
                 n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
    diff = np.array(ore) - np.array(baseline)
    n = len(diff)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(diff, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return np.mean(diff), lower, upper


def run_significance_tests(baseline: List[float], ore: List[float], 
                           metric_name: str = "NDCG@20") -> Dict:
    results = {
        "metric": metric_name,
        "n_queries": len(baseline),
        "baseline_mean": np.mean(baseline),
        "ore_mean": np.mean(ore),
        "improvement": np.mean(ore) - np.mean(baseline),
        "improvement_pct": ((np.mean(ore) - np.mean(baseline)) / np.mean(baseline)) * 100
    }
    
    t_stat, t_pval = paired_t_test(baseline, ore)
    results["t_test"] = {"t_statistic": t_stat, "p_value": t_pval}
    
    w_stat, w_pval = wilcoxon_test(baseline, ore)
    results["wilcoxon"] = {"statistic": w_stat, "p_value": w_pval}
    
    mean_diff, ci_lower, ci_upper = bootstrap_ci(baseline, ore)
    results["bootstrap"] = {
        "mean_diff": mean_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper
    }
    
    wins = sum(1 for b, o in zip(baseline, ore) if o > b)
    ties = sum(1 for b, o in zip(baseline, ore) if o == b)
    losses = sum(1 for b, o in zip(baseline, ore) if o < b)
    results["win_tie_loss"] = {"wins": wins, "ties": ties, "losses": losses}
    
    return results


def print_significance_report(results: Dict):
    print(f"\n{'='*60}")
    print(f"Statistical Significance Report: {results['metric']}")
    print(f"{'='*60}")
    
    print(f"\nDescriptive Statistics (n={results['n_queries']})")
    print(f"   Baseline Mean: {results['baseline_mean']:.4f}")
    print(f"   ORE Mean:      {results['ore_mean']:.4f}")
    print(f"   Improvement:   {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)")
    
    print(f"\nWin/Tie/Loss")
    wtl = results['win_tie_loss']
    print(f"   Wins: {wtl['wins']} | Ties: {wtl['ties']} | Losses: {wtl['losses']}")
    
    print(f"\nPaired t-test")
    t = results['t_test']
    sig = "SIGNIFICANT (p < 0.05)" if t['p_value'] < 0.05 else "Not significant"
    print(f"   t-statistic: {t['t_statistic']:.4f}")
    print(f"   p-value:     {t['p_value']:.4f} --> {sig}")
    
    print(f"\nWilcoxon Signed-Rank Test")
    w = results['wilcoxon']
    if np.isnan(w['p_value']):
        print(f"   Not enough non-tied pairs")
    else:
        sig = "SIGNIFICANT (p < 0.05)" if w['p_value'] < 0.05 else "Not significant"
        print(f"   statistic: {w['statistic']:.4f}")
        print(f"   p-value:   {w['p_value']:.4f} --> {sig}")
    
    print(f"\nBootstrap 95% Confidence Interval")
    b = results['bootstrap']
    contains_zero = b['ci_95_lower'] <= 0 <= b['ci_95_upper']
    sig = "Contains zero (not significant)" if contains_zero else "Does not contain zero (significant)"
    print(f"   Mean Diff: {b['mean_diff']:.4f}")
    print(f"   95% CI:    [{b['ci_95_lower']:.4f}, {b['ci_95_upper']:.4f}]")
    print(f"   Result:    {sig}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    
    # TREC-DL 2019 NDCG scores
    baseline_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.8436, 0.9238, 1.0, 0.8974, 0.4573, 0.9649,
        1.0, 1.0, 0.6522, 1.0, 0.7853, 1.0, 0.9357, 1.0, 0.7614, 1.0,
        0.5765, 1.0, 0.8672, 0.7951, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.6813, 1.0, 1.0, 0.8230, 1.0, 0.6272,
        0.6645, 0.7259, 1.0
    ]
    
    ore_2019 = [
        1.0, 0.8944, 1.0, 1.0, 0.9096, 0.9238, 1.0, 0.8974, 0.6085, 0.9649,
        1.0, 1.0, 0.7536, 1.0, 0.8391, 1.0, 0.9357, 1.0, 0.8355, 1.0,
        0.5765, 1.0, 0.8672, 0.8232, 0.9026, 1.0, 0.5765, 1.0, 0.9580, 0.9211,
        0.7361, 1.0, 1.0, 0.9498, 0.7221, 1.0, 1.0, 0.8230, 1.0, 0.6554,
        0.6645, 0.7571, 1.0
    ]
    
    # TREC-DL 2020 NDCG scores
    baseline_2020 = [
        0.8628, 1.0, 0.7410, 0.9134, 0.9505, 1.0, 0.6932, 0.5765, 1.0, 0.7620,
        0.9421, 1.0, 0.6915, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.2505, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7361,
        0.6858, 0.9802, 0.8193, 1.0
    ]
    
    ore_2020 = [
        0.8628, 1.0, 0.7876, 0.9134, 0.9011, 1.0, 0.7821, 0.5765, 1.0, 0.8196,
        0.9421, 1.0, 0.7449, 0.8830, 0.5539, 1.0, 0.8984, 1.0, 0.9802, 0.8895,
        0.8777, 0.8193, 0.3603, 1.0, 0.6565, 0.8820, 0.9131, 0.6168, 1.0, 0.9543,
        0.5765, 0.5539, 1.0, 0.8467, 0.7361, 0.6522, 1.0, 0.8375, 0.8777, 0.7361,
        1.0, 0.8777, 0.8649, 0.9131, 0.5539, 0.8131, 1.0, 0.5765, 0.9211, 0.7683,
        0.6858, 0.9802, 0.8534, 1.0
    ]
    
    print("\n" + "="*70)
    print("       STATISTICAL SIGNIFICANCE ANALYSIS FOR ORE")
    print("="*70)
    
    print("\nTREC-DL 2019")
    results_2019 = run_significance_tests(baseline_2019, ore_2019, "NDCG@20")
    print_significance_report(results_2019)
    
    print("\nTREC-DL 2020")
    results_2020 = run_significance_tests(baseline_2020, ore_2020, "NDCG@20")
    print_significance_report(results_2020)
    
    print("\nCOMBINED (Both Datasets)")
    combined_baseline = baseline_2019 + baseline_2020
    combined_ore = ore_2019 + ore_2020
    results_combined = run_significance_tests(combined_baseline, combined_ore, "NDCG@20")
    print_significance_report(results_combined)