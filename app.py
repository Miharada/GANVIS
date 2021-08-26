from flask import Flask, render_template, request
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
import numpy as np
from io import BytesIO
import base64

import pygad
import GenAlg as GA

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from werkzeug.utils import secure_filename

#https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/
UPLOAD_FOLDER = "Uploaded/"
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# port = int(os.environ.get("PORT", 5000))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home_page():
    init_generation = 50
    init_popsize = 10
    init_featnum = 2


    return render_template('base.html', 
                            generation=init_generation, 
                            popsize=init_popsize, 
                            featnum=init_featnum)

@app.route('/uploader', methods = ['GET', 'POST'])
def feat_select():
    if request.method == 'POST':
        models = request.form['model']
        generation = int(request.form.get('generations', False))
        popsize = int(request.form.get('popsize', False))
        featnum = int(request.form.get('featnum', False))
        f = request.files['file']
        if f and allowed_file(f.filename):
            file = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], file))
            model = model_selector(models)
            try:
                df = pd.read_csv(UPLOAD_FOLDER+f.filename)
            except:
                df = pd.read_excel(UPLOAD_FOLDER+f.filename)
            df.dropna(axis=0, inplace=True)
            labelencoder = LabelEncoder()
            # print(df.dtypes)
            for col in df.columns:
                if(not is_numeric_dtype(df[col])):
                    df[col] = df[col].astype(str)
                    df[col] = labelencoder.fit_transform(df[col])
            x,y = featureLabeldivider(df)
            
        featSelection = GA.GA(model=model, popsize=popsize, max_feat=featnum, iter_=generation)
        # print("Data",model, generation, popsize, featnum)
        mse, feat, max_acc, avg_acc_list = featSelection.fit(x,y)
        xaxis = [i for i in range(len(mse))]


        fig = plt.Figure()
        ax = fig.subplots()
        ax.plot(xaxis, avg_acc_list)
        # ax.plot(xaxis, avg_acc_list)
        buf = BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")


    return render_template('feature_select_result.html', hasilgraph=data,
                            model = models,
                            generation=generation,
                            popsize=popsize,
                            featnum=featnum,
                            feature=feat,
                            mse=mse,
                            max_acc=max_acc,
                            avg_acc_list=avg_acc_list,
                            xaxis=xaxis)

@app.route('/clusteringhome')
def clusterhome():
    clusters = 2
    generation = 10
    popsize = 10
    parentmating = 5
    keepparent = 2
    initlow = -2
    initmax = 2


    return render_template("clustering.html", 
                            clusters = clusters,
                            generation = generation,
                            popsize = popsize,
                            parentmating = parentmating,
                            keepparent = keepparent,
                            initlow = initlow,
                            initmax = initmax)

# Cluster 
data = [0]
num_cluster = 0
num_genes = 0
feature_vector_length = 0
@app.route('/clust', methods = ['GET', 'POST'])
def clustering():
    global data, num_cluster, num_genes, feature_vector_length
    pca = PCA(n_components=2)
    if request.method == 'POST':
        num_cluster = int(request.form.get('cluster', False))
        generation = int(request.form.get('generations', False))
        popsize = int(request.form.get('popsize', False))
        parentmating = int(request.form.get('parentmating', False))
        keepparent = int(request.form.get('keepparent', False))
        minumber = int(request.form.get('initlow', False))
        maxnumber = int(request.form.get('initmax', False))

        f = request.files['file']
        if f and allowed_file(f.filename):
            file = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], file))
           
            try:
                df = pd.read_csv(UPLOAD_FOLDER+f.filename)
            except:
                df = pd.read_excel(UPLOAD_FOLDER+f.filename)
            df.dropna(axis=0, inplace=True)
            labelencoder = LabelEncoder()
            # print(df.dtypes)
            for col in df.columns:
                if(not is_numeric_dtype(df[col])):
                    df[col] = df[col].astype(str)
                    df[col] = labelencoder.fit_transform(df[col])
            principalComponents = pca.fit_transform(df)
            principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
            data = principalDf.to_numpy()
            feature_vector_length = data.shape[1]
            num_genes = feature_vector_length * num_cluster
    
    ga_instance = pygad.GA(num_generations=100,
                sol_per_pop=10,
                init_range_low=-2,
                init_range_high=2,
                num_parents_mating=5,
                keep_parents=2,
                num_genes=num_genes,
                fitness_func=fitness_func,
                suppress_warnings=True)

    ga_instance.run()
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist = cluster_data(best_solution, best_solution_idx)

    fig = plt.Figure()
    ax = fig.subplots()
    for cluster_idx in range(num_cluster):
        cluster_x = data[clusters[cluster_idx], 0]
        cluster_y = data[clusters[cluster_idx], 1]
        ax.scatter(cluster_x, cluster_y)
        ax.scatter(cluster_centers[cluster_idx, 0], cluster_centers[cluster_idx, 1], linewidths=5)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    # plt.title("Clustering using PyGAD")
    # plt.show()

    return render_template("clustering_result.html", hasilgraph=data,
                            clusters = num_cluster,
                            generation = generation,
                            popsize = popsize,
                            parentmating = parentmating,
                            keepparent = keepparent,
                            initlow = minumber,
                            initmax = maxnumber)
# Cluster

def euclidean_distance(X, Y):
    return np.sqrt(np.sum(np.power(X - Y, 2), axis=1))


def cluster_data(solution, solution_idx):

    global num_cluster, feature_vector_length, data
    cluster_centers = [] 
    all_clusters_dists = [] 
    clusters = [] 
    clusters_sum_dist = [] 


    for clust_idx in range(num_cluster):
        cluster_centers.append(solution[feature_vector_length*clust_idx:feature_vector_length*(clust_idx+1)])
        cluster_center_dists = euclidean_distance(data, cluster_centers[clust_idx])
        all_clusters_dists.append(np.array(cluster_center_dists))

    cluster_centers = np.array(cluster_centers)
    all_clusters_dists = np.array(all_clusters_dists)

    cluster_indices = np.argmin(all_clusters_dists, axis=0)
    for clust_idx in range(num_cluster):
        clusters.append(np.where(cluster_indices == clust_idx)[0])
        
        if len(clusters[clust_idx]) == 0:
          
            clusters_sum_dist.append(0)
        else:
           
            clusters_sum_dist.append(np.sum(all_clusters_dists[clust_idx, clusters[clust_idx]]))
           

    clusters_sum_dist = np.array(clusters_sum_dist)

    return cluster_centers, all_clusters_dists, cluster_indices, clusters, clusters_sum_dist

def fitness_func(solution, solution_idx):
    _, _, _, _, clusters_sum_dist = cluster_data(solution, solution_idx)

    # The tiny value 0.00000001 is added to the denominator in case the average distance is 0.
    fitness = 1.0 / (np.sum(clusters_sum_dist) + 0.00000001)

    return fitness
   

def model_selector(modelselect):
    if modelselect == "SVC":
        return SVC()
    elif modelselect == "RFC":
        return RandomForestClassifier()
    elif modelselect == "AC":
        return AdaBoostClassifier()
    elif modelselect == "DTC":
        return DecisionTreeClassifier()

def featureLabeldivider(data):
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    return x,y

if __name__ == "__main__":
    app.run()
