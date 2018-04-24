import logging
from sklearn.decomposition import PCA

def pca_report(U_k):
    X = U_k
    pca = PCA(n_components=3)
    pca.fit(X)

    out_str = ''
    for i, (perc, comp) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):

        comp_string = ', '.join(['{:.1f}'.format(i) for i in comp])
        out_str += '\nComponent [{}] has weight {:.2f}'.format(comp_string, perc)
    return out_str

def spatial_weights_report(model):
    out_str = ''
    spatial_embeddings = [nmp.spatial_embedding for nmp in model.nmp_blocks] + [model.final_spatial_embedding]

    for i, spatial_embedding in enumerate(spatial_embeddings):
        U_k = spatial_embedding.weight.t().data.numpy()
        out_str += 'Layer {}'.format(i)
        out_str += pca_report(U_k)
        out_str += '\n'
        logging.info(out_str)

    return out_str
