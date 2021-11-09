from modeling.utils.models import *
from modeling.utils.data import *
from modeling.utils.sigma_vae import *
from modeling.utils.config import *
from modeling.utils.plotting import *

dt = build_data()
dt = weighted_anomaly(dt)

pivot_anomaly = flat_table(dt)

if reduction == "PCA":
    reduced_anomaly = reduce_dim(pivot_anomaly, method='PCA', exp_variance=0.999)
    folder = 'pca'
else:
    reduced_anomaly = reduce_dim(dt, method='VAE', season = season)
    folder = 'vae'

eofs, pcs = eofs(pivot_anomaly)
#pcs_df = pd.DataFrame(data=pcs, columns=['PC{}'.format(i) for i in range(1, pcs.shape[1]+1)],
#                      index = pivot_anomaly.time.values) #PCs are columns
#pcs_df.to_csv("pcs.csv", columns =['PC1','PC2'])
#plot_PC(pcs_df, col_name = 'PC1', savefig = True)
#plot_PC(pcs_df, col_name = 'PC2', savefig = True)
#plot_density("2021-01-01", "2021-02-28", pcs_df['PC1'], pcs_df['PC2'])
#plot_EOFS(eofs, savefig = False)

train_X, test_X, pivot_train, pivot_test = train_test_split(reduced_anomaly, pivot_anomaly, test_size = 0.2, random_state = 42)

if training:
    print("Starting cross-validation")
    for scoring in  ["score", "ch", "bic", "silhouette"]:
        estimator = cross_val(reduced_anomaly.values, method = model, scoring = scoring)
else:
    estimator = load_estimator(f'../models/{season}/{folder}/{model}_model_silhouette.pkl')

outputs = extract_regimes(train_X, method=model, nb_regimes = None, estimator = estimator)

if model == 'kmeans':
    labels, inertias, _ = outputs
elif model == 'bayesian_gmm' or model == 'gmm':
    probas, elbo, means, covariances, _ = outputs
    labels = np.argmax(probas, axis=1)
    plot_KL(means, covariances)
    plot_mixtures(reduced_anomaly.values, labels, means, covariances)

plot_regimes(pivot_anomaly, labels)