from modeling.utils.data import *
import os
from modeling.utils.models import *
#from modeling.utils.plotting import *
import matplotlib.pyplot as plt


#### Parameters
path = "P:\CH\Weather Data\ERA-5\GEOPOTENTIAL"
file = "ERA-5_hourly_Geopotential-500hPa_1-24Nov2021.nc"
file_normal = "ERA-5_daily_Geopotential-500hPa_DecJanFeb_1979-2020_normal.nc"


# ####### 1 - READING DATA OF NOV 2021

### Conversion Hourly -> Daily
file_inf = read_nc(os.path.join(path,file))
file_inf_daily = hourly_to_daily(file_inf)
file_inf_daily = limit_geography(file_inf_daily, LAT, LONG)
file_inf_daily /= G

# Read Normal File
z_normal = read_nc(os.path.join(path,file_normal))

z_anomaly = file_inf_daily - z_normal
z_anomaly = weighted_anomaly(z_anomaly)

####### 1 - READING DATA FULL PERIOD
#file_full = "ERA-5_daily_Geopotential-500hPa_DecJanFeb_1979-2020_anomaly.nc"
#z_anomaly_full = read_nc(os.path.join(path,file_full))



# Flat the array (from 3D to 2D)
pivot_anomaly = flat_table(z_anomaly)
reduced_pca = reduce_dim(pivot_anomaly,method="PCA",exp_variance=15)
#reduced_vae = reduce_dim(z_anomaly_full,method="VAE",season="WINTER",model="sigma_vae_statedict_5")

### Inference with PCA
model_pca = load_estimator("../models/WINTER/pca/kmeans_model_ch.pkl")
reduced_pca = np.array(reduced_pca, dtype=np.double)
pca_inference = model_pca.predict(reduced_pca)
pca_inference_onehot = np.zeros((pca_inference.size, pca_inference.max()+1))
pca_inference_onehot[np.arange(pca_inference.size),pca_inference] = 1
df = pd.DataFrame(pca_inference_onehot)
df.rename(columns =  {0: "SB", 1: "NAO-", 2: "NAO+", 3: "AR"}, inplace = True)
df.plot(kind='line')
plt.show()

# kmeans - plots of centroids
outputs = extract_regimes(reduced_pca, method='kmeans', nb_regimes = None, estimator = model)
labels, inertias, _ = outputs
plot_regimes(pivot_anomaly, labels)



### Inference with VAE - Bayesian GMM
model_vae = load_estimator("../models/WINTER/vae/bayesian_gmm_model_bic.pkl")
# df = pd.DataFrame(model_vae.predict_proba(reduced_vae))
# df.rename(columns =  {0: "SB", 1: "NAO-", 2: "AR", 3: "NAO+"}, inplace = True)
# df.plot(kind='line')

probas, elbo, means, covariances, _ = extract_regimes(reduced_vae, method='bayesian_gmm', nb_regimes = 5, estimator = model_vae)
labels_baygmm = np.argmax(probas, axis=1)
plot_regimes(pivot_anomaly, labels_baygmm)



### Inference with VAE - GMM
model_vae_gmm = load_estimator("../models/WINTER/vae/gmm_model_ch.pkl")
# df = pd.DataFrame(model_vae_gmm.predict_proba(reduced_vae))
# df.rename(columns =  {0: "NAO-", 1: "SB", 2: "AR", 3: "NAO+"}, inplace = True)
# df.plot(kind='line')

probas, elbo, means, covariances, _ = extract_regimes(reduced_vae, method='gmm', nb_regimes = 5, estimator = model_vae_gmm)
labels_gmm = np.argmax(probas, axis=1)
plot_regimes(pivot_anomaly, labels_gmm)

