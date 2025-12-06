import pandas as pd


features = pd.read_csv("/Users/lorenzgregorsauthoff/Downloads/fma_metadata/features.csv")


feature_cols = [
    "mfcc_01_mean", "mfcc_02_mean", "mfcc_03_mean", "mfcc_04_mean", "mfcc_05_mean",
    "mfcc_06_mean", "mfcc_07_mean", "mfcc_08_mean", "mfcc_09_mean", "mfcc_10_mean",
    "rms_mean",
    "tempo",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "chroma_var"
]


df_small = features[["track_id"] + feature_cols].copy()

df_small.to_csv("features_small.csv", index=False)

print("Compression complete! New file saved as features_small.csv")
