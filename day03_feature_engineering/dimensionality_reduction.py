import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def run_pca_tsne():
    try:
        df = pd.read_csv('titanic_cleaned.csv')
    except:
        print("Error: Run feature_engineering.py first!")
        return

    y = df['survived']
    X = df.drop(columns=['survived'])

    # 1. PCA (Principal Component Analysis)
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    # 3. Save Visualization
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='coolwarm', alpha=0.7)
    plt.title('PCA Projection')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='coolwarm', alpha=0.7)
    plt.title('t-SNE Clustering')

    plt.savefig('dim_reduction_plot.png')
    print("Visualization saved as 'dim_reduction_plot.png'")

if __name__ == "__main__":
    run_pca_tsne()
