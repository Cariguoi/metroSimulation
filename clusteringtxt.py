'''
Module for clustering on text coordinates.

This module is made of functions for computing the clustering
regions based on coordinates(left, top, height, width) of texts from
a page(web, image, ...).


The main functions:  clustering_text_results, clusters_coordinates

- Input(parameters): - filename: None or a 'filename'(e.g: 'data.pkl' file) 
                     - dict_txt: None or dictionary,
                     - x_colname & y_colname: x and y columns name from the data for the kmeans cluster plot(x and y axis, string),
                     - pca(True or False): if we use pca or not(default: False); 
                     - n_components: the number of components for the pca[if used](default: None[n_components = nb_columns_data]),
                     - useful_col: None or a list of columns names of the data to use(if None: [useful_col = columns_name_data]),
                     - init_method: kmeans method to use (default: 'k-means++'), 
                     - nb_clusters: the number of clusters to keep(None or an integer; if None: the optimal number of cluster is compute using elbow curve method)
                     - fig_name: name of the kmeans clusters plot to save(default: 'kmeans_clusters_text_with'), 
                     - cluster_data_name: name of the clustered data to save(default:'clustered_data'), 
                     - margin: margin to compute the rectangle coordinates of cluster( default: 10),
                     - path: the path to create/save the directory results(default: './'),
                     - dir_name: the name of the directory to save the results[images and data](default: 'clustering_results').
                        
        

- Output(results): - clustered dataset(initial dataset merged with clusters labels)
                   - cluters_results: dictionary of clusters coordinates(x_pt_origin, y_pt_origin, height, width)[forming a rectangle around each cluster group]
                                      with lenght and text of each cluster.
                   - images of clustering results(saved in a specified directory)
                   - .csv files(datasets saved in a specified directory) 

Methods: - kmeans: is a clustering method which aim is to partition n observations into k clusters(here, the number k of cluster is compute using elbow curve method).
                   Clusters group are obtained by minimizing within-cluster variances(or maximizing the sum of squared deviations between observation in different clusters).
        
         - pca(principal component anlysis): is a method which aim at reducing the dimensions of the given dataset while still retaining most of its variance. 
                                             It will reduce the  dimension of dataset by conserving the most important informations(variable with higher explained variances)


'''

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numbers

# to save the results
def create_dir(dir_name = "clustering_rslts"):
    #  create a directory if not existed 

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    else:
        sys.exit('Please rename the directory for your clustering results with the "dir-name" parameter of clustering_text_results function. \n A directory with the name ' + dir_name + ' existed already.')
    

# functions for clustering        

def load_data(filename = 'filename.pickle'):
    ''' load a pickle file.'''

    with open(filename, 'rb') as f:
        unserialized_data = pickle.load(f)
    
    return unserialized_data


def get_data(dict_txt):
    ''' get dataset from dictionary.'''
    
    return pd.DataFrame(dict_txt)


def clean_data(data):
    ''' clean the dataset. '''

    # convert data to pd.dataFrame
    data = pd.DataFrame(data)

    len_input_data = len(data)

    # print('\n the input data before cleaning:')
    # print(data.head(10))

    # replace empty rows with 'NaN' value
    data.replace(to_replace='', value=np.nan, inplace=True)

    # drop rows which contain empty spaces in 'text' column
    data.dropna(subset = ["text"], inplace=True)
    
    nb_droped_rows = len_input_data - len(data)
    # print('number of rows of input data:', len_input_data)
    # print('number of rows removed from input data:', nb_droped_rows)
    # print('number of rows of cleaned data:', len(data))

    return data


def get_x_vect(data, useful_col = None):
    ''' get x vector(with numerical values) for kmeans clustering. '''

    if useful_col != None:
        x_vect = data.loc[:,useful_col]
    else:
        x_vect = data
    
    return x_vect.select_dtypes(include=np.number)


def scale_data(data):
    ''' let each features of the dataset(with numerical values) have a mean of 0 and standard deviation of 1. '''

    # get data with numerical values
    data_num = data.select_dtypes(include=np.number)

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_num.values), columns=data_num.columns, index=data_num.index)
    
    return scaled_data


def get_pca(data, n_components = None, path = './'):
    ''' compute principal components analysis of a dataset(numerical values).'''

    pca = PCA(n_components = n_components)
    principal_components = pca.fit_transform(data)

    # build and save the explained variances plot:
    components = range(1, pca.n_components_+1)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(components, pca.explained_variance_ratio_.cumsum(), marker='o', color='blue')
    ax.set_xlabel('Number of components', fontsize=11)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=11)
    ax.set_title('Explained Variance by Components', fontsize=11)

    # define the path of the directory to save the figure
    fig_name = 'pca_explained_variance.png'
    fig_name_path = os.path.join(path, fig_name)  

    # save the figure
    fig.savefig(fig_name_path)

    # Save components to a DataFrame
    pca_comp_name = ['component '+ str(i) for i in components]
    pca_components = pd.DataFrame(principal_components, columns=pca_comp_name)
    pca_data = pd.concat([data, pca_components], axis = 1)

    # define the path of the directory to save the figure
    pca_data_name = 'pca_components_data.csv'
    pca_data_name_path = os.path.join(path, pca_data_name)    

    # export data into csv
    pca_data.to_csv (pca_data_name_path, index=None, header = True)

    return pca_components




def get_nb_clusters(data, path = './'):
    ''' get the optimal number of clusters using elbow curve(build and save in the current directory) method.
        data: dataset (with numerical values) for kmeans clustering.
     '''

    # define a range of values of k from 1 to 10
    K_clusters = range(1,11)

    # for each value of k, run k-means clustering
    kmeans = [KMeans(n_clusters=i) for i in K_clusters]

    # for each value of k, calculate the Sum of Squared Errors (SSE)[here score ]
    score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]
    
    # compute the optimal number of clusters
    kneedle = KneeLocator(x=K_clusters, y=score, S=1.0, curve="concave", direction="increasing")
    nb_clusters = kneedle.elbow

    # build and save the elbow curve:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(K_clusters, score, 'bo-')
    ax.set_xlabel('Number of Clusters', fontsize=11)
    ax.set_ylabel('Sum of Squared Errors', fontsize=11)
    

    # get column names of the data
    col_name = list(data.columns)

    # to reduce the size of the file name
    if len(col_name) > 3:
        col_name = [str(len(col_name)),'variables']

    # define the figure title with useful variables and the nb of clusters
    fig_title = ['Elbow Curve(']
    fig_title.extend(col_name)
    fig_title = ' '.join(fig_title) + ') with ' + str(nb_clusters) + ' as optimal number of clusters '
    ax.set_title(fig_title, fontsize=11)

    # define the figure name with useful variables and the nb of clusters
    fig_name = ['elbow_curve']
    fig_name.extend(col_name)
    fig_name = '_'.join(fig_name)
    fig_name = fig_name +  '_and_' + str(nb_clusters) + '_' + 'clusters' + '.png'

    # define the path of the directory to save the figure
    fig_name_path = os.path.join(path, fig_name)    

    # save the figure
    fig.savefig(fig_name_path)
   
    return nb_clusters


def get_rect_coord(x, y, margin):
    ''' compute rectangle coordinates to contain a scatter plot given a margin and (x,y)'''

    min_x, max_x = min(x) - margin, max(x) + margin
    min_y, max_y = min(y) - margin, max(y) + margin
    width = max_x - min_x
    height = max_y - min_y

    #save results(point of origin coords, width, height)
    coord_list = [min_x, min_y, width, height]

    return coord_list


def add_rectangle(current_plot, coord_list):
    ''' draw rectangle arround scatter plot using the above 
        coordinates(point of origin coords, width, height)
    '''
    
    # define each coordinate
    x = coord_list[0]
    y = coord_list[1]
    width = coord_list[2]
    height = coord_list[3]

    # draw the rectangle
    current_plot.add_patch(
        patches.Rectangle(
        xy=(x, y),  # point of origin.
        width=width, height=height, linewidth=1,
        color='red', fill=False))





def get_clust_coord(clustered_data, x_colname, y_colname, margin, path = './'):
    ''' compute the coordinates(point of origin coord(x,y), width, height
        to draw a rectangle around each cluster
    '''
    
    #initialise list and dictionaries to get the results
    clust_coord_dict = {}
    cluster_label_list = []
    x_pt_origin_rect_list = []
    y_pt_origin_rect_list = []
    width_rect_list = []
    height_rect_list = []
    length_cluster_list = []
    text_cluster_list = []

    clust_rslts = {}
    cluster_labels = clustered_data['cluster_label'].unique()

    for cluster_nb in cluster_labels:
        data_clust = clustered_data.loc[clustered_data['cluster_label'] == cluster_nb ]
        x_clust = data_clust[x_colname]
        y_clust = data_clust[y_colname]

        coord_list_clust = get_rect_coord(x = x_clust, y = y_clust, margin = margin)
        cluster_name = 'cluster ' + str(cluster_nb)
        clust_coord_dict[cluster_name] = coord_list_clust

        # get all results in list: coordinates, text; rect means rectangle(draw arround cluster)
        cluster_label_list.append(cluster_nb)
        x_pt_origin_rect_list.append(coord_list_clust[0])
        y_pt_origin_rect_list.append(coord_list_clust[1])
        width_rect_list.append(coord_list_clust[2])
        height_rect_list.append(coord_list_clust[3])
        length_cluster_list.append(len(data_clust['cluster_label']))
        text_cluster_list.append(data_clust['text'].tolist())

    # get all results in list: coordinates, text; rect means rectangle(draw arround cluster)
    clust_rslts['cluster_label'] = cluster_label_list
    clust_rslts['x_pt_origin_rect'] = x_pt_origin_rect_list
    clust_rslts['y_pt_origin_rect'] = y_pt_origin_rect_list
    clust_rslts['width_rect'] = width_rect_list
    clust_rslts['height_rect'] = height_rect_list
    clust_rslts['length_cluster'] = length_cluster_list
    clust_rslts['text_cluster'] = text_cluster_list


    # get the data from the dictionaries 'clust_coord_dict', 'clust_rslts'
    # clust_coord_data = pd.DataFrame(clust_coord_dict)
    clust_rslts_data = pd.DataFrame(clust_rslts)

    # define the path of the directory to save the data
    # cluster_coord_name_path = os.path.join(path, 'clusters_coordinates.csv') 
    cluster_rslts_name_path = os.path.join(path, 'clusters_results.csv') 

    # export data into csv
    # clust_coord_data.to_csv (cluster_coord_name_path, index=None, header = True)
    clust_rslts_data.to_csv (cluster_rslts_name_path, index=None, header = True)
    
    return clust_coord_dict, clust_rslts




def get_clusters(data, x_vect_kmeans, useful_col, x_colname, y_colname, margin,
                 nb_clusters = 2, init_method = 'k-means++',
                 fig_name = 'kmeans_clusters_text_with',
                 cluster_data_name = 'clustered_data', path = './'
                ):

    ''' compute kmeans clustering of numerical columns of a given data set. '''

    # define kmeans arguments
    kmeans = KMeans(n_clusters = nb_clusters, init = init_method)

    # define kmeans vector
    X = x_vect_kmeans

    # compute k-means clustering
    kmeans.fit(X) 

    # compute cluster centers and predict cluster index for each sample.
    data['cluster_label'] = kmeans.fit_predict(X)

    # get coordinates of cluster centers.
    centers = kmeans.cluster_centers_ 

    # predict the closest cluster each sample in X belongs to.
    labels = kmeans.predict(X) 

    if x_colname is None or y_colname is None:
        sys.exit('Please, x_colname and x_colname must begin given. \n They are the name of the column data corresponding to x, y axis for the plot')

    # build and save the results(plot the data colored by these labels)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    x_value = data[x_colname]
    y_value = data[y_colname]
    ax.scatter(x = x_value, y = y_value, c=labels, s=15)

    
    # compute coordinates list of rectangle for each cluster
    clust_coord_dict = get_clust_coord(clustered_data = data, 
                                        x_colname = x_colname, 
                                        y_colname = y_colname, 
                                        margin = margin,
                                        path = path
                                        )[0]

    # add rectangle arround each cluster group
    for cluster in clust_coord_dict.keys():
        coord_list_clust = clust_coord_dict[cluster]
        add_rectangle(current_plot = ax, coord_list = coord_list_clust)   
    

    #x = sns.scatterplot(x = data.left, y = data.top, c=labels, s=50, hue=data.cluster_label)
    #x = plt.scatter(centers[:, 0], centers[:, 1], c='black', s=20, alpha=0.5)
    
    # invert axis of not to get the same position as in the image
    #ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlabel('left', fontsize=11)
    ax.set_ylabel('top', fontsize=11)
    

    # to reduce the size of the file name
    if len(useful_col) > 3:
        useful_col = [str(len(useful_col)),'variables']

    # define the figure title with useful variables and the nb of clusters
    fig_title = ['kmeans clusters on text coordinates(']
    fig_title.extend(useful_col)
    fig_title = ' '.join(fig_title) + ') with nb_clusters = ' + str(nb_clusters) 
    #x = plt.title(fig_title, fontsize=11)
    ax.set_title(fig_title, fontsize=11)

    # define the figure name with useful variables and the nb of clusters
    fig_name = [fig_name]
    fig_name.extend(useful_col)
    fig_name = '_'.join(fig_name)
    fig_name = fig_name +  '_and_' + str(nb_clusters) + '_' + 'clusters' + '.png'
    
    # define the path of the directory to save the figure
    fig_name_path = os.path.join(path, fig_name)    
    
    # save the figure
    fig.savefig(fig_name_path)

    # define the clustered data name
    cluster_data_name = [cluster_data_name]
    cluster_data_name.extend(useful_col)
    cluster_data_name = '_'.join(cluster_data_name)
    cluster_data_name = cluster_data_name + '_with_' + str(nb_clusters) + '_' + 'clusters' + '.csv'

    # define the path of the directory to save the data
    cluster_data_name_path = os.path.join(path, cluster_data_name)    

    # export data into csv
    data.to_csv (cluster_data_name_path, index=None, header = True)

    return data





def clustering_text_results(x_colname, y_colname, dict_txt = None, filename = None,
                            pca = False, n_components = None,
                            useful_col = None, init_method = 'k-means++', 
                            nb_clusters = None, fig_name = 'kmeans_clusters_text_with', 
                            cluster_data_name = 'clustered_data', margin = 10,
                            path = './', dir_name = 'clustering_results'
                            ):
    
    ''' compute kmeans clustering from a .pkl file or dictionary using the above functions. '''

    print('\n ====== clustering on text coordinates - start ======')

    if filename is not None:
        print('\n ======= text clustering coordinates from pickle file =======')
        print('\n ======= load the pickle file =======')
        dict_txt = load_data(filename = filename)

    elif dict_txt is not None:
        print('\n ======= text clustering coordinates from dictionary =======')
    else:
        sys.exit('\n please, add as data input, either a .pkl filename or a dictionary(dict_txt) to use this function.')

    
    print('\n ======= get the dataset from the dictionary =======')
    data_txt = get_data(dict_txt)

    print('\n ======= clean the dataset =======')
    cleaned_data = clean_data(data_txt)

    # create the directory to save the clustering results
    create_dir(dir_name = dir_name )
    path = path + dir_name

    if useful_col is None:
        print('\n ======= getting the columns names of the vector for kmeans clustering =======' )
        useful_col = cleaned_data.columns.values.tolist()
    
    print('\n ======= get the vector for kmeans clustering =======' )
    x_vect = get_x_vect(cleaned_data, useful_col)

    print('\n ======= scale the vector for kmeans clustering =======')
    x_vect_scaled = scale_data(x_vect)
    

    if pca:
        print('\n ======= using pca: compute principal components, the vector for kmeans clustering =======')
        x_vect_kmeans = get_pca(data = x_vect_scaled, n_components = n_components, path = path)

        print('\n The first 5 rows of the pca data of principal components:')
        print(x_vect_kmeans.head(5))  

    else:
        x_vect_kmeans = x_vect_scaled

    
    if nb_clusters is None or not isinstance(nb_clusters, numbers.Number):
        print('\n ======= get the number of clusters =======')
        nb_clusters = get_nb_clusters(x_vect_kmeans, path = path)
   
    print('\n ======= compute the kmeans clusters =======')
    clustered_data  = get_clusters(data = cleaned_data, x_vect_kmeans = x_vect_kmeans, useful_col = useful_col,
                                         nb_clusters = nb_clusters, init_method = init_method,
                                         x_colname = x_colname, y_colname = y_colname, margin = margin, 
                                         fig_name = fig_name, cluster_data_name = cluster_data_name, path = path
                                        )


    print('\n ======= compute the clusters coordinates dictionary results =======')
    clust_rslts_dict = get_clust_coord(clustered_data = clustered_data , x_colname = 'left', 
                            y_colname = 'top', margin = margin, path = path)[1]

    print('\n The first 5 rows of the clustered data:')
    print(clustered_data.head(5))  

    print('\n The rectangle coordinates of each cluster:')
    print(clust_rslts_dict)                                  
    

    print('\n \n ======= clustering done: the results are save in', path,'   ======')

    return clustered_data, clust_rslts_dict





def clusters_coordinates(x_colname, y_colname, dict_txt = None, filename = None,
                            pca = False, n_components = None,
                            useful_col = None, init_method = 'k-means++', 
                            nb_clusters = None, fig_name = 'kmeans_clusters_text_with', 
                            cluster_data_name = 'clustered_data', margin = 10,
                            path = './', dir_name = 'clustering_results'
                            ):
    
    ''' compute coordinates of clusters from kmeans clustering and from a .pkl file or dictionary using the above functions.
        NB: second version of the clustering_text_results, without print messages and return only cluster coordinates.
    '''


    if filename is not None:
        dict_txt = load_data(filename = filename)

    elif dict_txt is None:
        sys.exit('\n please, add as data input, either a .pkl filename or a dictionary(dict_txt) to use this function.')

    
    data_txt = get_data(dict_txt)
    cleaned_data = clean_data(data_txt)

    # create the directory to save the clustering results
    create_dir(dir_name = dir_name )
    path = path + dir_name

    if useful_col is None:
        useful_col = cleaned_data.columns.values.tolist()
    
    x_vect = get_x_vect(cleaned_data, useful_col)
    x_vect_scaled = scale_data(x_vect)
    

    if pca:
        x_vect_kmeans = get_pca(data = x_vect_scaled, n_components = n_components, path = path)
    else:
        x_vect_kmeans = x_vect_scaled

    
    if nb_clusters is None or not isinstance(nb_clusters, numbers.Number):
        nb_clusters = get_nb_clusters(x_vect_kmeans, path = path)
   
    clustered_data  = get_clusters(data = cleaned_data, x_vect_kmeans = x_vect_kmeans, useful_col = useful_col,
                                         nb_clusters = nb_clusters, init_method = init_method,
                                         x_colname = x_colname, y_colname = y_colname, margin = margin, 
                                         fig_name = fig_name, cluster_data_name = cluster_data_name, path = path
                                        )


    clust_rslts_dict = get_clust_coord(clustered_data = clustered_data , x_colname = 'left', 
                            y_colname = 'top', margin = margin, path = path)[1]
    

    print('\n \n   clustering done: the results are save in', path,'   \n \n')

    return clust_rslts_dict


