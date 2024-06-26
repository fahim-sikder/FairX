from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd
import numpy as np
import time

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as FF

import plotly.express as plotly_express


def visualize_tsne(ori_data, fake_data, save_fig = False):

    """
    Visual metrics to see the similarity between data distributions. Used PCA and t-SNE plots.

    Input: ori_data: numpy array, original data
            fake_data: numpya array, synthetic data
            save_fig: Boolean, if true, the plot will be saved
    """
    
    ori_data = np.asarray(ori_data)

    fake_data = np.asarray(fake_data)
    
    ori_data = ori_data[:fake_data.shape[0]]
    
    sample_size = 64
    
    idx = np.random.permutation(len(ori_data))[:sample_size]
    
    randn_num = np.random.permutation(sample_size)[:1]
    
    real_sample = ori_data[idx]

    fake_sample = fake_data[idx]
    
    real_sample_2d = real_sample
    
    fake_sample_2d = fake_sample
        

        
    ### PCA
    
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pca_real._append(pca_synthetic).rename(
        columns={0: '1st Component', 1: '2nd Component'})
    
    
    ### TSNE
    
    tsne_data = np.concatenate((real_sample_2d,
                            fake_sample_2d), axis=0)

    tsne = TSNE(n_components=2,
                verbose=0,
                perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    
    
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'
    
    fig, axs = plt.subplots(ncols = 2, nrows=1, figsize=(10, 5))

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs[0])
    sb.despine()
    
    axs[0].set_title('PCA Result')


    sb.scatterplot(x='X', y='Y',
                    data=tsne_result,
                    hue='Data', 
                    style='Data', 
                    ax=axs[1])
    sb.despine()

    axs[1].set_title('t-SNE Result')


    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)

    if save_fig:
    
        plt.savefig(f'{time.time():.2f}-tsne.png', dpi = 300)

## Plotting Intersectional bias

def plot_intersectional_bias(df, sensitive_attr = [], target_attr = None, save_fig = False):

    """
    Plot Intersectional bias in a dataset.
    
    Input: df: pandas.DataFrame
           sensitive_attr: list of protected attributes
           target_attr: target attribute from df
           save_fig: boolean, if True, plot will be saved
    """

    m = sb.FacetGrid(df, row = sensitive_attr[0], col = sensitive_attr[1])
    
    m.map(sb.histplot, target_attr, discrete = True, shrink = .8)
    
    plt.tight_layout()

    if save_fig:
    
        plt.savefig(f'{time.time():.2f}-ib.png', dpi = 300)
    
    plt.show()


## 3d plot to show the fairness vs utility

def make_3d_plot_fairness_utility(plot_df, x_axis_feature, y_axis_feature, z_axis_feature, save_fig = False):

    """
    Plot Fairness vs Data utility in different Methods;

    Input: plot_df: Pandas Dataframe, containing information about methods, and metrics
            x_axis_feature: Dataframe column name,
            y_axis_feature: Dataframe column name,
            z_axis_feature: Dataframe column name,
            save_fig: Boolean, if True, figure will be saved
    """

    figure = plotly_express.scatter_3d(plot_df,
                     x=x_axis_feature, y=y_axis_feature,
                     z=z_axis_feature, color="Methods", 
                     title="Fairness vs Data Utility")
     
    figure.update_layout(showlegend=True)

    if save_fig:

        figure.write_image(f'{time.time():.2f}-benchmarking-fig.png')
     
    figure.show()


def show_generated_imgs(imgs, img_name, save_fig = False):
    """
    Show generated or real images using torchvision grid.
    
    Input: imgs, torchvision grid
            img_name: string, file name to save the figure
            save_fig: Boolean, if true, figure will be saved
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize = (15, 9))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if save_fig:
        plt.savefig(f'{img_name}.png', dpi=300, transparent = True)