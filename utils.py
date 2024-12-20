import pickle

with open("models/pca.pkl", "wb") as f:
    pickle.dump(pca, f)
