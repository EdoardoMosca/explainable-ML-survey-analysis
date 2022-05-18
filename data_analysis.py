import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from Data import Data
# import umap
# import umap.plot
# import hdbscan

def calculate_correlation(df, data, name=None):
    """ Calculate the Pearson correlation for a set of variables"""
    cmap = sns.diverging_palette(240, 25, as_cmap=True)
    corr = df.corr() # compute pearson correlation 
    # create seabvorn heatmap with required labels
    c = sns.heatmap(corr, vmin=-1.0, vmax=1.0, cmap=cmap) #xticklabels=x_axis_labels, yticklabels=y_axis_labels,
  
    # Create plot
    plt.rc('figure', titlesize=16) # fontsize of figure label
    plt.plot()
    plt.suptitle("Data Pearson Correlation")
    cbar = c.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/Data_Pearson_Correlation.png') 
        plt.savefig('imgs/Data_Pearson_Correlation.svg') 

# def cluster_embeddings(embeddings, name=None):

#     # embeddings = umap.UMAP(n_neighbors=15, 
#     #                         n_components=10, 
#     #                         random_state=42,
#     #                         metric='cosine').fit_transform(embeddings)
#     cluster = hdbscan.HDBSCAN(min_cluster_size=10,
#                           metric='euclidean',                      
#                           cluster_selection_method='eom').fit(embeddings)

#     umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.5, random_state=42, metric='cosine').fit_transform(embeddings)
#     result = pd.DataFrame(umap_data, columns=['x', 'y'])
#     result['labels'] = cluster.labels_
#     fig, ax = plt.subplots(figsize=(5, 5))
#     outliers = result.loc[result.labels == -1, :]
#     clustered = result.loc[result.labels != -1, :]
#     plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.5)
#     plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.5, cmap='rainbow')
#     plt.colorbar()
#     plt.plot()
#     plt.suptitle("Topic Modelling with Embeddings")

#     if name: 
#         plt.savefig(name +".png") 
#     else:
#         plt.savefig('Topic_Modelling_with_Embeddings.png') 

# def visualize_label_for_embeddings(embeddings, df, data, bin=False, name=None):
#     labels = df.loc[:, data.label_columns[:]]

#     if bin==True:
#         labels = data.bin_labels(labels)

#     embeddings_2dim = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.5, random_state=42, metric='cosine').fit_transform(embeddings)
#     result = pd.DataFrame(embeddings_2dim, columns=['x', 'y'])

    
#     fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12,6), sharey=True, sharex=True) #
#     bp = []
#     axs_l = [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]]
#     for i, column in enumerate(data.label_columns):
#         #result['labels'] = labels.iloc[:,i]
#         bp.append(axs_l[i].scatter(result.x, result.y, c=labels.iloc[:,i], s=0.5, cmap='rainbow'))
#         plt.colorbar(bp[i], ax=axs_l[i])
#         plt.plot()
#         plt.suptitle("")

#     axs_l[0].set_title('For "employee small business/startup"', fontsize=6, fontweight="bold")
#     axs_l[1].set_title('For "employee SME/large business"', fontsize=6, fontweight="bold")
#     axs_l[2].set_title('For "employee non-profit organization"', fontsize=6, fontweight="bold")
#     axs_l[3].set_title('For "employee government, military, or public agency"', fontsize=6, fontweight="bold")
#     axs_l[4].set_title('For "teacher or educational professional K-12 school"', fontsize=6, fontweight="bold")
#     axs_l[5].set_title('For "faculty member/professional college/university"', fontsize=6, fontweight="bold")
#     axs_l[6].set_title('For "founding non profit"', fontsize=6, fontweight="bold")
#     axs_l[7].set_title('For "founding for profit"', fontsize=6, fontweight="bold")

#     plt.suptitle("Student Embeddings with Career label")

#     if name: 
#         plt.savefig(name +".png") 
#     else:
#         plt.savefig('Student_Embeddings_with Career_label.png') 


def build_wordcloud(text, name=None):
    """ 
    Plot a cloud of words 
    Parameters: 
        text (np.array): array containing text string, we want to analyze
    """

    X = text.astype(str)
    X = np.asarray(X)

    #Create wordcloud
    stopwords = set(STOPWORDS) 
    wordcloud = WordCloud(width = 800, height = 300, stopwords = 
                            stopwords,background_color ='white',  min_font_size = 10)
    all_text=""
    for x in X: 
        print(x)
        all_text += " " + x
    wordcloud.generate(all_text)

    # Create plot
    plt.figure(figsize = (8, 3), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/word_cloud.png')


def plot_label_correlations(df, data, name=None):
    """ 
    Plots the correlations between labels
    Parameters: 
        df (DataFrame): label dataframe of size N x 8
        data (object): data object from Data class
    """
    sns.set_theme()
    # Create correlation plot
    g = sns.PairGrid(df.loc[:, data.label_columns[:]], diag_sharey=False, corner=True)
    g.map_diag(sns.histplot, discrete=True)
    g.map_offdiag(sns.histplot, discrete=True)
    
    # Adjust and createplot
    g.add_legend()
    plt.plot()
    plt.suptitle("Label Correlation")

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/Label_Correlation.png') 



def plot_text_length_statistics(df, data, name=None):
    """ 
    Plots statistics for the text length
    Parameters: 
        df (DataFrame): text dataframe of size N x 1/2
        data (object): data object from Data class
    """

    #Set plot style parameters
    colors = ["#0065BD", "#A2AD00", "#64A0C8", "#E37222", "#005293"]
    sns.set_style(style="darkgrid", rc= {'patch.edgecolor': '#FFFFFF'})

    # Compute text lentgth and store it in additional column 
    column = 0
    df['text_length'] = df.loc[:, data.text_columns[column]].str.count(' ') + 1
    
    # Create plot with text length
    fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(23, 8), sharey=True) #
    
    # Create first figure for overall text length over the predictions
    bp = []
    bp.append(sns.boxenplot(y='text_length', data=df, ax=axs[0], showfliers=False, linewidth=0, color="#0065BD"))
    bp[0] = sns.stripplot(y='text_length', data=df, size=4, color="#0065BD", ax=axs[0], linewidth=0.3, edgecolor='#FFFFFF', dodge=True, alpha=.3, zorder=1)
    
    #Adjust plot style
    for line in bp[0].get_lines():
            line.set_color('white')
            line.set_linewidth(3)
        
    bp[0].set_xlabel('', fontsize=10)
    bp[0].set_ylabel('', fontsize=10)
    bp[0].tick_params(axis="x", labelsize=17) 
    bp[0].tick_params(axis="y", labelsize=17) 
    bp[0].plot()
    plt.suptitle("")
    
    # Append other figures for text length over the predictions for each prediction head
    for i, label_column in enumerate(data.label_columns):
        bp.append(sns.boxenplot(x=label_column, y='text_length', data=df, ax=axs[i+1], showfliers=False,  linewidth=0, palette=sns.color_palette(colors)))
        bp[i+1] = sns.stripplot(x=label_column, y='text_length', data=df, size=4, palette=sns.color_palette(colors), jitter=False, ax=axs[i+1], linewidth=0.3, edgecolor='#FFFFFF', dodge=False, alpha=.3, zorder=1)
        for line in bp[i+1].get_lines():
            line.set_color('white')
            line.set_linewidth(3)
        
        #Create labels
        bp[i+1].set_xlabel('', fontsize=10)
        bp[i+1].set_ylabel('', fontsize=10)
        bp[i+1].tick_params(axis="x", labelsize=17) 
        bp[i+1].tick_params(axis="y", labelsize=17) 
        bp[i+1].plot()
        plt.suptitle("")

    # Add titles
    fontsize_subtitle=18
    bp[0].set_title('Overall \n text length \n distribution', fontsize=fontsize_subtitle)
    bp[1].set_title('Employee \nsmall business/ \nstartup \n(L1)', fontsize=fontsize_subtitle)
    bp[2].set_title('Employee SME/ \nlarge business \n(L2)', fontsize=fontsize_subtitle)
    bp[3].set_title('Employee \nnon-profit \norganization \n(L3)', fontsize=fontsize_subtitle)
    bp[4].set_title('Employee \ngovernment/military/ \npublic agency \n(L4)', fontsize=fontsize_subtitle)
    bp[5].set_title('Teacher/educational \nprofessional \nin K-12 school \n(L5)', fontsize=fontsize_subtitle)
    bp[6].set_title('Faculty member/ \nprofessional in  \ncollege/university \n(L6)', fontsize=fontsize_subtitle)
    bp[7].set_title('Founding \nnon-profit \n(L7)', fontsize=fontsize_subtitle)
    bp[8].set_title('Founding \nfor-profit \n(L8)', fontsize=fontsize_subtitle)

    fig.text(0.01, 0.4, 'Text Length (number of words)', ha='center', va='center', fontsize=20, rotation='vertical')
    fig.text(0.5, 0.02, 'Aspiration level (0 to 4)', ha='center', va='center', fontsize=20)
    fig.text(0.5, 0.93, 'Text Length (Q Inspire) per Label Class', ha='center', va='center', fontsize=22)
    plt.tight_layout(pad=3)

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/text_length.png') 
        plt.savefig('imgs/text_length.svg') 


def plot_Q20_statistics(df, data, kind='hist', name=None):
    """ 
    Plots statistics for the class distribution of the different prediction heads
    Parameters: 
        df (DataFrame): label dataframe of size N x 8
        data (object): data object from Data class
        kind (string): kind of plot
    """
    # Set sns theme
    sns.set()
    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(23, 5), sharey=True, sharex=True) #
    bp = []

    #Create box or history plots for the label distributions 
    # accross the different classes for each prediction head
    for i, column in enumerate(df.columns):
        if kind=='box':
            bp.append(df.boxplot(column=[column], grid=False, fontsize=6, ax=axs[i]))

        if kind=='hist':
            bp.append(sns.histplot(df[column], stat="density", discrete=True, ax=axs[i], color="#64A0C8"))
            left, middle, right = np.percentile(np.asarray(df[column]), [25, 50, 75])
            axs[i].vlines(middle, 0, 0.6, color='#0065BD', ls=':')
            axs[i].vlines(left, 0, 0.4, color="#E37222", ls=':')
            axs[i].vlines(right, 0, 0.4, color="#E37222", ls=':')
        
        # Add labels
        bp[i].set_xlabel('', fontsize=10)
        bp[i].set_ylabel('', fontsize=10)
        bp[i].tick_params(axis="x", labelsize=17) 
        bp[i].tick_params(axis="y", labelsize=17) 
        bp[i].plot()
        plt.suptitle("")

    # Add titles
    fontsize_subtitle=18
    bp[0].set_title('Employee \nsmall business/ \nstartup \n(L1)', fontsize=fontsize_subtitle)
    bp[1].set_title('Employee SME/ \nlarge business \n(L2)', fontsize=fontsize_subtitle)
    bp[2].set_title('Employee \nnon-profit \norganization \n(L3)', fontsize=fontsize_subtitle)
    bp[3].set_title('Employee \ngovernment/military/ \npublic agency \n(L4)', fontsize=fontsize_subtitle)
    bp[4].set_title('Teacher/educational \nprofessional \nin K-12 school \n(L5)', fontsize=fontsize_subtitle)
    bp[5].set_title('Faculty member/ \nprofessional in  \ncollege/university \n(L6)', fontsize=fontsize_subtitle)
    bp[6].set_title('Founding \nnon-profit \n(L7)', fontsize=fontsize_subtitle)
    bp[7].set_title('Founding \nfor-profit \n(L8)', fontsize=fontsize_subtitle)

    if kind=='box':
        fig.text(0.5, 0.05, 'Aspiration level (0 to 4)', ha='center', va='center', fontsize=20, rotation='vertical')
    if kind=='hist':
        fig.text(0.5, 0.05, 'Aspiration level (0 to 4)', ha='center', va='center', fontsize=20)
        fig.text(0.01, 0.4, 'Density', ha='center', va='center', fontsize=20, rotation='vertical')

    fig.text(0.5, 0.93, 'Label Class Distributions (c=2)', ha='center', va='center', fontsize=22)
    plt.tight_layout(pad=3)

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/label_count.png') 
        plt.savefig('imgs/label_count.svg') 

def plot_Q20_pred_label_distr(df, data, name=None):
    """ 
    Plots statistics for the prediction versus label distribution for the different prediction heads
    Parameters: 
        df (DataFrame): label dataframe of size N x 8
        data (object): data object from Data class
    """
    # Set sns theme
    sns.set()
    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(23, 5), sharey=True, sharex=True) #
    bp = []
    axs_d = []

    #Create history plots for the label distributions and continuous or history plots for the predicitons
    # accross the different classes for each prediction head
    for i in range(len(data.label_columns)):
        bp.append(sns.histplot(df['l'+str(i+1)], stat="density", alpha = 0.7, discrete=True, ax=axs[i], color="#64A0C8"))
        axs_d.append(axs[i].twinx())
        # for classification 
        bp[i] = sns.histplot(np.rint(np.asarray(df['p'+str(i+1)])).astype(np.int32), stat="density", alpha = 0.4, discrete=True, ax=axs_d[i], color="#E37222")
        #for regression
        #bp[i] = sns.kdeplot(df['p'+str(i+1)],  fill=True, alpha = 0.4, ax=axs_d[i], color="#E37222")

        # Add percentile lines
        left, middle, right = np.percentile(np.asarray(df['l8']), [25, 50, 75])
        axs[i].vlines(middle, 0, 0.6, color='#0065BD', ls=':')
        axs[i].vlines(left, 0, 0.4, color="#E37222", ls=':')
        axs[i].vlines(right, 0, 0.4, color="#E37222", ls=':')
        
        # Add labels
        axs[i].set_xlabel('', fontsize=10)
        axs[i].set_ylabel('', fontsize=10)
        axs_d[i].set_xlabel('', fontsize=10)
        axs_d[i].set_ylabel('', fontsize=10)
        axs[i].tick_params(axis="x", labelsize=17) 
        axs[i].tick_params(axis="y", labelsize=17) 
        axs[i].set_ylim(0, 1)
        axs_d[i].set_yticklabels([])
        axs_d[i].grid(False)
        axs_d[i].set_ylim(0, 1)
        plt.tick_params(right=False)
        bp[i].plot()
        plt.suptitle("")

    # Add titles
    fontsize_subtitle=18
    bp[0].set_title('Employee \nsmall business/ \nstartup \n(L1)', fontsize=fontsize_subtitle)
    bp[1].set_title('Employee SME/ \nlarge business \n(L2)', fontsize=fontsize_subtitle)
    bp[2].set_title('Employee \nnon-profit \norganization \n(L3)', fontsize=fontsize_subtitle)
    bp[3].set_title('Employee \ngovernment/military/ \npublic agency \n(L4)', fontsize=fontsize_subtitle)
    bp[4].set_title('Teacher/educational \nprofessional \nin K-12 school \n(L5)', fontsize=fontsize_subtitle)
    bp[5].set_title('Faculty member/ \nprofessional in  \ncollege/university \n(L6)', fontsize=fontsize_subtitle)
    bp[6].set_title('Founding \nnon-profit \n(L7)', fontsize=fontsize_subtitle)
    bp[7].set_title('Founding \nfor-profit \n(L8)', fontsize=fontsize_subtitle)
    fig.text(0.5, 0.05, 'Aspiration level (0 to 4)', ha='center', va='center', fontsize=20)
    fig.text(0.008, 0.4, 'Density', ha='center', va='center', fontsize=20, rotation='vertical')
    fig.text(0.5, 0.93, 'Prediction vs label distribution', ha='center', va='center', fontsize=22)
    plt.tight_layout(pad=3)

    #Save plot
    if name: 
        plt.savefig("{}.png".format(name)) 
        plt.savefig("{}.svg".format(name)) 
    else:
        plt.savefig('imgs/label_prediction_distribution.png') 
        plt.savefig('imgs/label_prediction_distribution.svg') 


def plot_bias_var_labels(df_labels, data, name=None):
    """ 
    Plots bias and variance for the labels accross label heads for samples
    Parameters: 
        df_labels (DataFrame): label dataframe of size N x 8
        data (object): data object from Data class
    """
    #Compute mean and std for the samples accross labels
    df_labels['mean'] = df_labels.mean(axis=1)
    df_labels['std'] = df_labels.std(axis=1)

    #Set sns theme 
    sns.set()
    sns.color_palette("rocket")

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharey=True, sharex=True) #

    # Create hist plot for mean and std
    bp1 = sns.histplot(df_labels['mean'], kde=True, stat="density", discrete=True, ax=axs[0])
    bp1.set_title('Bias', fontsize=6, fontweight="bold")
    bp1.set_xlabel('label', fontsize=10)
    bp1.plot()

    bp2 = sns.histplot(df_labels['std'], kde=True, stat="density", discrete=True, ax=axs[1])
    bp2.set_title('Std', fontsize=6, fontweight="bold")
    bp2.set_xlabel('label', fontsize=10)
    bp2.plot()
    plt.suptitle('Label bias and std of individuals', fontsize=14)

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/bias_std.png') 

def TF_IDF(df, data, label, name=None):
    """ 
    Compute Term_frequency_Inverse_Document_Frequency 
    Parameters: 
        df (DataFrame): dataset dataframe 
        data (object): data object from Data class
        label (string): label we want to look at
    """
    # Compute
    labels = df.loc[:, data.label_columns[:]]
    # Standardize labels to account for individual bia
    if data.account_bias==True:
        labels = data.standardize_individuals(labels)
    

    labels = np.asarray(labels.iloc[:, label])

    # Get TF_IDF
    X_text = df.loc[:, data.text_columns[:]].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_df=800, min_df=10)
    Xtr = vectorizer.fit_transform(X_text)
    features = vectorizer.get_feature_names()

    # Plot top feats
    dfs = top_feats_by_class(Xtr, labels, features, min_tfidf=0.1, top_n=10)
    plot_tfidf_classfeats_h(dfs, name)


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs, name):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/TF_IDF.png') 


def keyword_confusion(df, data, name=None):
    """ 
    Calculate a confusion matrix showing which label the samples containing certain keywords 
    indicating a high label value belong to
    Parameters: 
        df (DataFrame): dataset dataframe 
        data (object): data object from Data class
    """
    #Set theme
    cmap = sns.diverging_palette(240, 25, as_cmap=True)
   
    #Keywords
    l1_kw = '|'.join(['job', 'small', 'companies', 'company', 'small business','join a start-up', 'join a startup', 'work for a startup', 'work for a start-up', 'small company'])
    l2_kw = '|'.join(['job', 'companies', 'company' 'industry','medium sized business', 'medium business','large business', 'established company', 'medium sized company', 'medium company', 'large company'])
    l3_kw = '|'.join(['work for a non-profit','work for a nonprofit','society', 'social'])
    l4_kw = '|'.join([ 'officer', 'Army', 'NASA', 'Nasa', 'Navy', 'Naval', 'Air Force', 'public', 'national', 'government','military', 'public agency'])
    l5_kw = '|'.join(['teacher','teach at school', 'K12', 'high school', 'college'])
    l6_kw = '|'.join(['PhD','phd','phd', 'professor', 'work at university', 'lab', 'research'])
    l7_kw = '|'.join(['start my own','found my own', 'my own nonprofit', 'my own non-profit', 'not some profiteering business'])
    l8_kw = '|'.join(['entrepreneur', 'ventures', 'found', 'my own company', 'my own business','my own startup', 'build my own', 'create my own', 'start a business', 'start' ])

    keywords = [l1_kw, l2_kw, l3_kw, l4_kw, l5_kw, l6_kw, l7_kw, l8_kw]
    correl = []

    # Count keywords in text and append sample if it contans keyword
    for i in range(len(keywords)):
        df['text_l'+str(i+1)] = df[data.text_columns[0]].str.contains(keywords[i], case=False).astype(int)
        correl.append(df.groupby(label_columns[i])['text_l'+str(i+1)].value_counts().unstack().fillna(0))

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(23, 5), sharey=True, sharex=True) #
    bp = []
    for i, column in enumerate(data.label_columns):
        # Create heatmap and set labels
        bp.append(sns.heatmap(correl[i]/correl[i].to_numpy().sum(), annot=True, annot_kws={"fontsize":14}, fmt='.1%', ax=axs[i], cmap=cmap))
        bp[i].set_xlabel('', fontsize=10)
        bp[i].set_ylabel('', fontsize=10)
        bp[i].tick_params(axis="x", labelsize=17) 
        bp[i].tick_params(axis="y", labelsize=17) 
        cbar = bp[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=17)
        bp[i].plot()
        plt.suptitle("")

    # Add title
    fontsize_subtitle=18
    bp[0].set_title('Employee \nsmall business/ \nstartup \n(L1)', fontsize=fontsize_subtitle)
    bp[1].set_title('Employee SME/ \nlarge business \n(L2)', fontsize=fontsize_subtitle)
    bp[2].set_title('Employee \nnon-profit \norganization \n(L3)', fontsize=fontsize_subtitle)
    bp[3].set_title('Employee \ngovernment/military/ \npublic agency \n(L4)', fontsize=fontsize_subtitle)
    bp[4].set_title('Teacher/educational \nprofessional \nin K-12 school \n(L5)', fontsize=fontsize_subtitle)
    bp[5].set_title('Faculty member/ \nprofessional in  \ncollege/university \n(L6)', fontsize=fontsize_subtitle)
    bp[6].set_title('Founding \nnon-profit \n(L7)', fontsize=fontsize_subtitle)
    bp[7].set_title('Founding \nfor-profit \n(L8)', fontsize=fontsize_subtitle)

    fig.text(0.5, 0.03, 'Keywords contained in text', ha='center', va='center', fontsize=20)
    fig.text(0.01, 0.4, 'Label', ha='center', va='center', fontsize=20, rotation='vertical')
    fig.text(0.5, 0.93, 'Label-Text Correlation (Q Inspire)', ha='center', va='center', fontsize=22)
    plt.tight_layout(pad=3)

    #Save plot
    if name: 
        plt.savefig("imgs/{}.png".format(name)) 
        plt.savefig("imgs/{}.svg".format(name)) 
    else:
        plt.savefig('imgs/correlation_text_label.png') 
        plt.savefig('imgs/correlation_text_label.svg') 

def plot_confusion_matrix(df, data, name=None):
    """ 
    Calculate a confusion matrix for the predictions vs the labels
    Parameters: 
        df (DataFrame): dataframe of prediction/ label dataset
        data (object): data object from Data class
    
    """
    # set the theme
    cmap = sns.light_palette('#005293', as_cmap=True)

    y_true = np.zeros((df.shape[0],len(data.label_columns)))
    y_pred = np.zeros((df.shape[0],len(data.label_columns)))

    # Create column names
    columns_pred = []
    columns_label = [] 
    for i in range(len(data.label_columns)):
        columns_pred.append('p'+str(i+1))
        columns_label.append('l'+str(i+1))

    # Add labels and predictions to an array containing all heads
    for i, label in enumerate(columns_label):
        y_true[:, i] = np.asarray(df[label], dtype=np.int32)

    for i, pred in enumerate(columns_pred):
        y_pred[:, i] = np.rint(np.asarray(df[pred])).astype(np.int32)

    labels = [i for i in range(len(np.unique(y_true)))]
    index = ''.join(str(i) for i in labels)
    columns = ''.join(str(i) for i in labels)

    # Create figure
    fig, axs = plt.subplots(nrows=1, ncols=8, figsize=(23, 5.2), sharey=True, sharex=True, gridspec_kw={'width_ratios':[1,1,1,1,1,1,1,1.20]}) #
    bp = []

    # Create and plot confusion matrix for each head
    for i in range(len(data.label_columns)):
        confusion = confusion_matrix(y_true[:,i], y_pred[:,i], labels=labels)

        #Claculating metrics: 
        accuracy = np.diagonal(confusion).sum()/confusion.sum()
        precision = []
        recall = []
        f1 = []
        for n in range(confusion.shape[0]):
            precision_n = confusion[n,n]/(confusion[:,n].sum())
            precision.append(precision_n)
            recall_n = confusion[n,n]/(confusion[n,:].sum())
            recall.append(recall_n)
            f1.append(2 * (precision_n * recall_n) / (precision_n + recall_n))
        print("Metrics for head {}: accuracy={}, precision={}, recall={}, f1score={}".format(i,accuracy,precision, recall, f1 ))
        
        confusion = pd.DataFrame(confusion, index = [x for x in index],
                    columns = [c for c in columns])
        
        #Add detailed plot descriptions
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_percentages = ['{0:.1%}'.format(value) for value in
                        confusion.to_numpy().flatten()/confusion.to_numpy().sum()]
        plot_labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
        plot_labels = np.asarray(plot_labels).reshape(2,2)
        
        bp.append(sns.heatmap(confusion/confusion.to_numpy().sum(), annot=plot_labels, annot_kws={"fontsize":14}, fmt='', ax=axs[i], cbar=i == 7,
                vmin=0, vmax=0.65, cmap=cmap))
        bp[i].set_xlabel('', fontsize=10)
        bp[i].set_ylabel('', fontsize=10)
        bp[i].tick_params(axis="x", labelsize=17) 
        bp[i].tick_params(axis="y", labelsize=17) 
        bp[i].plot()
        plt.suptitle("")

    #Add title
    fontsize_subtitle=18
    bp[0].set_title('Employee \nsmall business/ \nstartup \n(L1)', fontsize=fontsize_subtitle)
    bp[1].set_title('Employee SME/ \nlarge business \n(L2)', fontsize=fontsize_subtitle)
    bp[2].set_title('Employee \nnon-profit \norganization \n(L3)', fontsize=fontsize_subtitle)
    bp[3].set_title('Employee \ngovernment/military/ \npublic agency \n(L4)', fontsize=fontsize_subtitle)
    bp[4].set_title('Teacher/educational \nprofessional \nin K-12 school \n(L5)', fontsize=fontsize_subtitle)
    bp[5].set_title('Faculty member/ \nprofessional in  \ncollege/university \n(L6)', fontsize=fontsize_subtitle)
    bp[6].set_title('Founding \nnon-profit \n(L7)', fontsize=fontsize_subtitle)
    bp[7].set_title('Founding \nfor-profit \n(L8)', fontsize=fontsize_subtitle)
    cbar = bp[7].collections[0].colorbar
    cbar.ax.tick_params(labelsize=17)
    fig.text(0.5, 0.03, 'Prediction', ha='center', va='center', fontsize=20)
    fig.text(0.01, 0.4, 'Label', ha='center', va='center', fontsize=20, rotation='vertical')
    fig.text(0.5, 0.93, 'Confusion Matrix Prediction vs Labels', ha='center', va='center', fontsize=22)
    plt.tight_layout(pad=3)

    #Save plot
    if name: 
        plt.savefig("{}.png".format(name)) 
        plt.savefig("{}.svg".format(name)) 
    else:
        plt.savefig('imgs/confusion.png') 
        plt.savefig('imgs/confusion.svg') 


if __name__=="__main__":
    with open('Classification/constants.json') as f:
        constants = json.load(f)

    with open('Classification/dataset_variables.json') as d:
        dataset_variables = json.load(d)

    # Initialize label and feature columns
    label_columns = dataset_variables['label_columns']
    feature_columns = dataset_variables['feature_columns']
    text_columns = dataset_variables['text_columns']

    # Initialize Data Class
    data = data = Data(constants, dataset_variables) 
    # Load dataset
    df = pd.read_excel (r'./Data/EMS1.xlsx', sheet_name='EMS 1.0 All 7197 Items KH2 ')
    df = data.prepare_raw_data(df)

    #####################################

    ################# Label statistics
    #df_labels = df.loc[:, data.label_columns[:]]
    
    #plot_Q20_statistics(df_labels, data, 'hist', name='label_stats_4')
    
    # df_labels = data.standardize_individuals(df_labels)
    # #plot_bias_var_labels(df_labels, data, name='stand_label_bias_var')
    # plot_Q20_statistics(df_labels, data, 'hist', name='stand_label_stats')
    
    #df_bin_labels = data.bin_labels(df_labels)
    #plot_bias_var_labels(df_bin_labels, data, name='label_stats_2_med')
    #plot_Q20_statistics(df_bin_labels, data, 'hist', name='label_stats_2_higher_equal2')
    
    #plot_label_correlations(df, data)
    #calculate_correlation(df, data)

    ################## Text Plots
    #build_wordcloud(df.loc[:, data.text_columns[:]])
    #plot_text_length_statistics(df, data, name="text_length_statistics_Inspire")
    #TF_IDF(df, data, 0)
    #keyword_confusion(df, data)

    ################## Prediction Plots
    df_pred_train = pd.read_excel (r'Data/predictions_dataset.xlsx', sheet_name='train')
    df_pred_val = pd.read_excel (r'Data/predictions_dataset.xlsx', sheet_name='val')
    df_pred_test = pd.read_excel (r'Data/predictions_dataset.xlsx', sheet_name='test')

    plot_Q20_pred_label_distr(df_pred_test, data)
    plot_confusion_matrix(df_pred_test, data)

    ###################CLuster plots
    # #Get embeddings, label, and text from Excel file
    # df_emb = pd.read_excel(r'Data/EMS_embeddings_mean.xlsx', sheet_name='embeddings')
    # BERT_emb = np.asarray(data.prepare_X_emb(df_emb))

    # total_features = []
    # for topic_features in data.feature_columns.values():
    #     total_features += topic_features
    
    # feat_emb = np.asarray(df.loc[:, total_features])

    #cluster_embeddings(feat_emb, name="Topic Modelling with feat_topic5 Embeddings")
    #visualize_label_for_embeddings(feat_emb, df, data, name="Feat_topic5 Embeddings with Career label")

    #cluster_embeddings(BERT_emb, name="Topic Modelling with BERT Embeddings")
    #visualize_label_for_embeddings(BERT_emb, df_emb, data, name="BERT Embeddings with Career label")