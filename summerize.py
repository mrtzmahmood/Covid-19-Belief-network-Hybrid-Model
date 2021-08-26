import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

device='cpu'

engine = create_engine('sqlite:///data/covid_db.db')

def feature_extraction(text):
    x = tokenizer.encode(text)
    with torch.no_grad():
        x, _ = bert(torch.stack([torch.tensor(x)]).to(device))
        return list(x[0][0].cpu().numpy())
    
def data_prep(dataset):
    X = []
    for element in tqdm(dataset):
        if element['tokens']:
            X.append(feature_extraction(element['tokens']))
    return np.array(X)

def visualization_variables():
    data ={}
    df = pd.DataFrame(data)
    df_text = pd.read_csv('raw_data/labeled/labeled_data.csv')
    df_text = df_text.rename(columns = {df_text.columns[23]:'positive'})
    df_text = df_text.rename(columns = {df_text.columns[24]:'negative'})
    df_text = df_text.rename(columns = {df_text.columns[25]:'neutral'})
    df_text = df_text.rename(columns = {df_text.columns[50]:'lpositive'})
    df_text = df_text.rename(columns = {df_text.columns[51]:'lnegative'})
    df ['postId']  = df_text['Post Id']
    df ['pos'] = df_text['positive']
    df ['neg'] = df_text['negative']
    df ['neu'] = df_text['neutral']
    df ['lpos'] = df_text['lpositive']
    df ['lneg'] = df_text['lnegative']  
    return df
    
def bbn_summerized_variables():
    df_train = pd.read_sql('SELECT * FROM train',engine)
    df_train = df_train.drop('level_0', 1)
    df_train = df_train.drop('file_name', 1)
    df_train = df_train.drop('index', 1)
    df_train=df_train.rename(columns = {'textField_normal':'content'})
    df_train ['tokens']= df_train['tokens'].str.replace(r'"', '')
    df_train ['tokens']= df_train['tokens'].str.replace('|', ' ')
    df_train ['tokens']= df_train['tokens'].str[:512]
    df_train['X30'] = df_train['X30'].astype(int)
    le = preprocessing.LabelEncoder()
    le.fit(df_train['X1'])
    df_train['X1'] = le.transform(df_train['X1'])
    le.fit(df_train['X2'])
    df_train['X2'] = le.transform(df_train['X2'])
    le.fit(df_train['X3'])
    df_train['X3'] = le.transform(df_train['X3'])
    le.fit(df_train['X4'])
    df_train['X4'] = le.transform(df_train['X4'])
    le.fit(df_train['X5'])
    df_train['X5'] = le.transform(df_train['X5'])
    le.fit(df_train['X6'])
    df_train['X6'] = le.transform(df_train['X6'])
    le.fit(df_train['X7'])
    df_train['X7'] = le.transform(df_train['X7'])
    le.fit(df_train['X8'])
    df_train['X8'] = le.transform(df_train['X8'])
    le.fit(df_train['X9'])
    df_train['X9'] = le.transform(df_train['X9'])
    le.fit(df_train['X10'])
    df_train['X10'] = le.transform(df_train['X10'])
    le.fit(df_train['X11'])
    df_train['X11'] = le.transform(df_train['X11'])
    le.fit(df_train['X12'])
    df_train['X12'] = le.transform(df_train['X12'])
    le.fit(df_train['X13'])
    df_train['X13'] = le.transform(df_train['X13'])
    le.fit(df_train['X14'])
    df_train['X14'] = le.transform(df_train['X14'])
    le.fit(df_train['X15'])
    df_train['X15'] = le.transform(df_train['X15'])
    le.fit(df_train['X16'])
    df_train['X16'] = le.transform(df_train['X16'])
    le.fit(df_train['X17'])
    df_train['X17'] = le.transform(df_train['X17'])
    le.fit(df_train['X18'])
    df_train['X18'] = le.transform(df_train['X18'])
    le.fit(df_train['X19'])
    df_train['X19'] = le.transform(df_train['X19'])
    le.fit(df_train['X20'])
    df_train['X20'] = le.transform(df_train['X20'])
    le.fit(df_train['X21'])
    df_train['X21'] = le.transform(df_train['X21'])
    le.fit(df_train['X22'])
    df_train['X22'] = le.transform(df_train['X22'])
    le.fit(df_train['X23'])
    df_train['X23'] = le.transform(df_train['X23'])
    le.fit(df_train['X24'])
    df_train['X24'] = le.transform(df_train['X24'])
    le.fit(df_train['X25'])
    df_train['X25'] = le.transform(df_train['X25'])
    le.fit(df_train['X26'])
    df_train['X26'] = le.transform(df_train['X26'])
    le.fit(df_train['X27'])
    df_train['X27'] = le.transform(df_train['X27'])
    le.fit(df_train['X28'])
    df_train['X28'] = le.transform(df_train['X28'])
    le.fit(df_train['X29'])
    df_train['X29'] = le.transform(df_train['X29'])
    X1= df_train['X12']
    X3 = df_train['X29']
    c =0
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X26 ==1:
            c =1
    X4=[]
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X26==1 or row.X23==1 or row.X23==1:
            X4.append(0)
        elif ((row.X1==3 and row.X17==3) or (row.X24==1 and row.X17==3) or (row.X1==3 and row.X9==3)):
            X4.append(1)
        elif ((row.X21==1 or row.X1 ==1 or row.X25==1) or (row.X24==1 and row.X17==1)):
            X4.append(2)
        elif (row.X27==1 or (row.X1==3 and row.X9==1)):
            X4.append(3)
        else:
            X4.append(4)
    X5 =[]
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X12 ==3 and row.X13 ==1:
            X5.append(0)
        elif row.X3 == row.X8:
            X5.append(1)
        elif ((row.X3 ==1 or row.X8 ==1) and (row.X4 ==3 or row.X5 ==3 or row.X6==3 or row.X7 ==3)):
            X5.append(2)
        else:
            X5.append(3)
    X6 =[]
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X1==3 and row.X27==1:
            X6.append(0)
        elif ((row.X21==1 and row.X17==3) or (row.X21==1 and row.X9==3) or (row.X21==1 and row.X15==3)):
            X6.append(1)
        elif ((row.X21==1 and row.X17==1) or (row.X21==1 and row.X9==1) or (row.X21==1 and row.X15==1)):
            X6.append(2)
        elif row.X24 ==1 and row.X17 ==1:
            X6.append(3)
        else:
            X6.append(4)
    X7 = df_train['X9']
    X9 = df_train['X14']
    X10 = []
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X15==1 or row.X17==1 or row.X2 ==0:
            X10.append(0)
        elif row.X15==3 or row.X17==3 or row.X2 ==3:
            X10.append(1)
        else:
            X10.append(2)
    X11 = []
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X8==1 or row.X3==1 or row.X16==1 or row.X28==1:
            X11.append(0)
        elif row.X8==3 or row.X3==3 or row.X16==3 or row.X28==3:
            X11.append(1)
        else:
            X11.append(2)     
    X12 = []
    for row in df_train.itertuples(index=True, name='Pandas'):
        if row.X4 ==1 and row.X5==1 and row.X6==1 and row.X7==1:
            X12.append(3)
        elif row.X4 ==3 and row.X5==3 and row.X6==3 and row.X7==3:
            X12.append(2)
        elif row.X4 ==2 and row.X5==2 and row.X6==2 and row.X7==2:
            X12.append(1)
        else:
            X12.append(0)   
    X13 =df_train['X16']
    X14 = df_train['X13']
    df_train['NX1'] = X1
#    df['X2'] = X2
    df_train['NX3'] = X3
    df_train['NX4'] = X4
    df_train['NX5'] = X5
    df_train['NX6'] = X6
    df_train['NX7'] = X7
#    df['X8'] = X8
    df_train['NX9'] = X9
    df_train['NX10'] = X10
    df_train['NX11'] = X11
    df_train['NX12'] = X12
    df_train['NX13'] = X13
    df_train['NX14'] = X14
    return df_train

def bert_featues(df):
    bertL = data_prep(df.to_dict('records'))
    bertX = pd.DataFrame(bertL)
    return bertX

def bow_features(df):
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000)
    bowL = bow_vectorizer.fit_transform(df['tokens'])
    bowX = pd.DataFrame(bowL.toarray())
    return bowX

def tfidf_features(df):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000)
    tfidfL = tfidf_vectorizer.fit_transform(df['tokens'])
    tfidfX = pd.DataFrame(tfidfL.toarray())
    return tfidfX

def word_vector(model_w2v, tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
                         
            continue
    if count != 0:
        vec /= count
    return vec

def word2vec_features(df):
    tokenized_content = df['tokens'].apply(lambda x: x.split()) # tokenizing
    model_w2v = gensim.models.Word2Vec(
                tokenized_content,
                size=200, # desired no. of features/independent variables 
                window=5, # context window size
                min_count=2,
                sg = 1, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 2, # no.of cores
                seed = 34)

    model_w2v.train(tokenized_content, total_examples= len(tokenized_content), epochs=20)
    wordvec_arrays = np.zeros((len(tokenized_content), 200))
    for i in range(len(tokenized_content)):
        wordvec_arrays[i,:] = word_vector(model_w2v, tokenized_content[i], 200)
    wordvecX = pd.DataFrame(wordvec_arrays)
    return wordvecX 