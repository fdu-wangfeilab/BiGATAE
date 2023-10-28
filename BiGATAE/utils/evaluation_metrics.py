from sklearn.metrics import  adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score, completeness_score


def clustering_metrics(adata, target, pred): 
    """
    Evaluate clustering performance.

    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` x `n_vars`. Rows correspond to cells and columns to genes.
    target
        Key in `adata.obs` where ground-truth spatial domain labels are stored.
    pred
        Key in `adata.obs` where clustering assignments are stored.

    Returns
    -------
    ami
        Adjusted mutual information score.
    ari
        Adjusted Rand index score.
    homo
        Homogeneity score.
    nmi
        Normalized mutual information score.
    """ 
    ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
    print("ARI ", "%.6f"%ari)
    ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
    print("AMI ", "%.6f"%ami)
    nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
    print("NMI ", "%.6f"%nmi)
    homo = homogeneity_score(adata.obs[target], adata.obs[pred])
    print("Homo ", "%.6f"%homo)
    comp = completeness_score(adata.obs[target], adata.obs[pred])
    print("Comp ", "%.6f"%comp)
    # print(jaccard_score(adata.obs[target], adata.obs[pred], average='macro')) # jaccard_index
    # # print("ji ", "%.6f"%ji)

def clustering_metrics_mean(adata, target, pred, n):
    ari, ami, nmi, homo, comp = 0
    for i in range(n):
        ari += adjusted_rand_score(adata.obs[target], adata.obs[pred])
        ami += adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        nmi += normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        homo += homogeneity_score(adata.obs[target], adata.obs[pred])
        comp += completeness_score(adata.obs[target], adata.obs[pred])
    print("ARI ", format(ari/n,'.6f'))
    print("AMI ", format(ami/n,'.6f'))
    print("NMI ", format(nmi/n,'.6f'))
    print("Homo ", format(homo/n,'.6f'))
    print("Comp ", format(comp/n,'.6f'))

