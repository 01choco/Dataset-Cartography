from correlations.cosine import cosine_calculation
from correlations.pearson import pearson_calculation
from correlations.pearson_minmax_normalization import pearson_minmax_calculation
from correlations.kendalls_tau import kendalls_tau_calculation
from correlations.kl_divergence import kl_divergence_calculation

def run_correlation(cfg, dataset):
    if cfg.corr_method == "cosine":
        cosine_calculation(cfg, dataset)
    elif cfg.corr_method == "pearson":
        pearson_calculation(cfg, dataset)
    elif cfg.corr_method == "pearson-minmax":
        pearson_minmax_calculation(cfg, dataset)
    elif cfg.corr_method == "kendall":
        kendalls_tau_calculation(cfg, dataset)
    elif cfg.corr_method == "kl":
        kl_divergence_calculation(cfg, dataset)