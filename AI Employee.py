#!/usr/bin/env python
# coding: utf-8

# In[31]:


import warnings
warnings.filterwarnings('ignore')
#loading dependencies
import pandas as pd
import os
#Data Preprocessing
def load_data(filename):
    _, file_extension = os.path.splitext(filename)
    # Load the file based on the extension
    if file_extension == '.csv':
        return pd.read_csv(filename)
    elif file_extension == '.json':
        return pd.read_json(filename)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(filename)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")



def checking_data(df):
    na=df.isna().sum()
    info=df.info()
    description=df.describe()
    shape=df.shape
    return na,info,description,shape


#data cleaning and processing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder,KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing_cleaning(df):
    #splitting the data into features and target
    x=df.drop('Total',axis=1)
    y=df['Total']
    #separating numeriacal and categorical feature
    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = x.select_dtypes(include=['object', 'category']).columns.tolist()
    # Creating transformers for numerical and categorical features
    numeric_transformer=Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',StandardScaler())
    ])
    categorical_transformer=Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])
    # combining the transformers
    preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_transformer,numeric_features),
        ('cat',categorical_transformer,cat_features)
    ])
    #creaitng complete pipeline
    pipeline=Pipeline(steps=[
        ('preprocessor',preprocessor)
    ])
    #fitting the pipeline to data
    x_processed=pipeline.fit_transform(x)
    
    #splitting data into traing and test set
    return x_processed,y

def analysis_engine(x_processed,y):
    #Linear Regression
    lr=LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split(x_processed,y,test_size=0.2,random_state=42)
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    result=r2_score(y_pred,y_test)
    print("Linear Regression Results:",result)
    
    #Decision Trees
    dt=DecisionTreeClassifier()
    binning = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    df['Category'] = binning.fit_transform(df[['Total']]).astype(int) 
    bin_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    df['Category_Label'] = df['Category'].map(bin_labels)
    x=df[['Gold','Silver','Bronze']]
    y=df['Category']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=DecisionTreeClassifier(random_state=42)
    model.fit(x_train,y_train)
    y_preds=model.predict(x_test)
    print('Classfication Report:\n',classification_report(y_test,y_preds))
    #visualizing
    plt.figure(figsize=(10,7))
    plot_tree(model,feature_names=['Gold','Silver','Bronze'],class_names=[bin_labels[i] for i in range (len(bin_labels))],filled=True,rounded=True)
    
    #Kmeans Clustering
    x=df[['Gold','Silver','Bronze']]
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x)    
    kmeans=KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(x_scaled)
    #visualizing
    plt.figure(figsize=(20,10))
    for cluster in range(3):
        cluster_data=df[df['Cluster']==cluster]
        plt.scatter(cluster_data['Gold'],cluster_data['Silver'],s=cluster_data['Bronze']*10,label=f'Cluster{cluster}',cmap='viridis')
    '''s=cluster_data['Bronze'] * 10: The size of each point is proportional to the number of bronze medals,
    scaled by a factor of 10 to make the points more visible.'''
    for i in range(df.shape[0]):
        if df['Cluster'][i]!=0:
            plt.text(df['Gold'][i],df['Silver'][i],df['Country'][i],fontsize=9,ha='right')
    plt.title("K-Means Clustering")
    plt.xlabel("Gold Medals")
    plt.ylabel('Silver Medals')
    plt.legend()
    plt.colorbar(label='Cluster')
    plt.show()
    
    #Reports and grapghs
def generate_visualizations(df,output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_filtered=df[df['Total']>50] 
    df_filtered.plot(kind='bar',x='Country',y=['Gold','Silver','Bronze'],stacked=True)
    plt.title('Medals Count by country')
    #plt.xlabel('Country')
    plt.ylabel('Number of Medals')
    plt.savefig(f'{output_dir}/medals_count.png')
    plt.show()
    plt.close()
    #pie chart
    df_filtered['Total'].plot(kind='pie',labels=df['Country'],autopct='%1.1f%%')
    plt.title("Total Medals Distribution by country")
    plt.savefig(f'{output_dir}/total_medals_distribution.png')
    plt.show()
    plt.close()

def main_pipeline(df):
    filename = input("Enter file name:")
    df = load_data(filename)
    df.head()
    checking_data(df)
    x_processed,y=preprocessing_cleaning(df)
    analysis_engine(x_processed,y)
    generate_visualizations(df)
    


# In[32]:


main_pipeline(df)

