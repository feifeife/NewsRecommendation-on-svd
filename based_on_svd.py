
# coding: utf-8



import jieba
import jieba.analyse
import pandas as pd
import numpy as np
import re
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from datetime import datetime
import codecs
from sklearn import preprocessing
#from __future__ import division, print_function
from gensim import corpora,similarities,models



# import data

path='./data/'
stop_words_path = path+'stopwords/baidu_stopwords.txt'
train_data = pd.read_csv(path+'train.csv').dropna()
test_data = pd.read_csv(path+'test.csv').dropna()
train_data['newswords'] = train_data['title']+' '+train_data['content']
test_data['newswords'] = test_data['title']+' '+test_data['content']

# 构建id2index词典
# train
df_train_userid = train_data.drop_duplicates(subset=['userid'],keep='first')
df_train_userid.reset_index(inplace=True,drop=True)
df_train_newsid = train_data.drop_duplicates(subset=['newsid'],keep='first')
df_train_newsid.reset_index(inplace=True,drop=True)
userid2index_dict={}
newsid2index_dict={}
for index,row in df_train_userid.iterrows():
    userid2index_dict[row["userid"]] = index
for index,row in df_train_newsid.iterrows():
    newsid2index_dict[row["newsid"]] = index
# test
df_test_userid = test_data.drop_duplicates(subset=['userid'],keep='first')
df_test_userid.reset_index(inplace=True,drop=True) # 重置索引
df_test_newsid = test_data.drop_duplicates(subset=['newsid'],keep='first')
df_test_newsid.reset_index(inplace=True,drop=True)
userid2index_dict_test={}
newsid2index_dict_test={}
for index,row in df_test_userid.iterrows():
    userid2index_dict_test[row["userid"]] = index
for index,row in df_test_newsid.iterrows():
    newsid2index_dict_test[row["newsid"]] = index


# 生成user-news矩阵
def create_matrix(data,userid2index,newsid2index):
    user_num = len(data['userid'].drop_duplicates())  
    # data.userid.unique().shape[0]
    news_num = len(data['newsid'].drop_duplicates())
    # data.newsid.unique().shape[0]
    data_matrix = np.zeros((user_num, news_num))

    # 生成数据矩阵。user-news,看过就+1
    for row in data.itertuples():
        data_matrix[userid2index[getattr(row,"userid")], newsid2index[getattr(row,"newsid")]] += 1 #对应的是转换过的id
    return data_matrix

# train_data_matrix = create_matrix(train_data,userid2index_dict,newsid2index_dict)
# test_data_matrix = create_matrix(test_data,userid2index_dict_test,newsid2index_dict_test)

# 只保留中文
def chinese_remained(line):
    rule = re.compile("[^\u4e00-\u9fa5]")
    line = re.sub(rule, "", line)
    return line

# 建立id2keywords映射
def keywords(df_uniqueby_newsid):
    news_id2keywords = []
    for row in df_uniqueby_newsid.itertuples():
        keywords = " ".join(jieba.analyse.textrank(chinese_remained(getattr(row,"newswords")),topK=20))
        news_id2keywords.append([getattr(row,"newsid"),keywords]) #原来的newsid，顺序是index
    df_id2keywords = pd.DataFrame(np.array(news_id2keywords),columns=["newsid","keywords"])
    return df_id2keywords

# df_id2keywords = keywords(df_train_newsid)
# df_id2keywords_test = keywords(df_test_newsid)
# 保存
# df_id2keywords.to_csv("train_news_keywords.csv",index=False)
# df_id2keywords_test.to_csv("test_news_keywords.csv",index=False)


# svd分解，重构矩阵
def svd(train_data_matrix):
#U为用户主题分布，sigma为奇异值，V为物品主题分布
    U,sigma,VT = svds(train_data_matrix, k=20)
    sigma= np.diag(sigma)
    reconstruct_matrix=np.dot(np.dot(U, sigma), VT) 
    filter_nonread = train_data_matrix < 1e-8 
    filter_readed = train_data_matrix > 1e-8
    svd_nonread=reconstruct_matrix* filter_nonread
    svd_readed=reconstruct_matrix* filter_readed
    return (svd_nonread,svd_readed)


# 反转字典
def invert_dict(d):
    return dict(zip(d.values(), d.keys()))

# 计算相似度
def calsim(df_id2keywords, df_id2keywords_test,rec_num):
    # 生成语料库
    corpora_documents = df_id2keywords['keywords'].values.tolist() #顺序是index
    corpora_documents = [[key for key in news.split(" ")] for news in corpora_documents]
    # 从语料库生成字典
    dictionary = corpora.Dictionary(corpora_documents)
    dictionary.save('dictionary.txt')  # 保存生成的词典
    # dictionary = corpora.Dictionary.load('dictionary.txt')#加载
    # 对每个文档生成词袋
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    corpora.MmCorpus.serialize('corpuse.mm', corpus)  # 保存生成的语料
    # corpus=corpora.MmCorpus('corpuse.mm')#加载

    tfidf_model = models.TfidfModel(corpus)
    # tfidf_model.save('tfidf_model.tfidf')
    # tfidf_model = models.TfidfModel.load("tfidf_model.tfidf")

    corpus_tfidf = tfidf_model[corpus]
    corpus_tfidf.save("data.tfidf")

    similarity = similarities.MatrixSimilarity(corpus_tfidf,num_best = rec_num)
    similarity.save('similarity.index')
    # similarity = similarities.Similarity.load('similarity.index')

    # 对测试集中每条新闻，对应生成train中每个文档对他的相似度
    corpora_documents_test = df_id2keywords_test['keywords'].values.tolist()
    corpora_documents_test = [[key for key in news.split(" ")] for news in corpora_documents_test]

    corpus_test = [dictionary.doc2bow(text) for text in corpora_documents_test]
    corpus_test_tfidf = tfidf_model[corpus_test]
    corpus_test_tfidf.save("data_test.tfidf")

    test_similarity = [similarity[test] for test in corpus_test_tfidf] # 返回tuples(index, similarity)
    return test_similarity
# 计算热门新闻
def freqclicknews_rec(df_train_news_click,sim_news_inverse,newsid2index_dict,index2newsid_dict_test,total_num,old_num,new_num):
    rec_freq = list(df_train_news_click.head(old_num)['newsid']) #原本的id
    
    # new news
    df_sim_list = df_train_news_click.head(new_num)
    if new_num==0:
        per_num=0
    else:
        per_num= int((total_num - old_num) / new_num)
    # 在测试集中找与热门新闻相近的新闻
    for i in range(new_num):
        index = newsid2index_dict[df_sim_list['newsid'][i]]    
        df_rec_byfreq = pd.DataFrame([x for x in sim_news_inverse[index][:per_num]], columns=['newsid_test', 'similarity'])# 是index
        df_rec_byfreq.sort_values(by=['similarity'], inplace=True)
        df_rec_byfreq.reset_index(inplace=True,drop=True)
        for j in range(per_num):
            rec_freq.append(index2newsid_dict_test[df_rec_byfreq['newsid_test'][j]])    # news_id
    return rec_freq #真实的newsid

# 推荐

def recommend(train_data,test_data,newsid2index_dict,userid2index_dict,newsid2index_dict_test,total_num,old_num,new_num):
    # 为每个train里的老用户推荐svd没看过的新闻
    if new_num==0:per_num=0
    else:
        per_num= int((total_num - old_num) / new_num)
    user_num = len(train_data['userid'].drop_duplicates())  
    news_num = len(train_data['newsid'].drop_duplicates())
    
    rec_bysvd=np.zeros((user_num,old_num))
    index2newsid_dict = invert_dict(newsid2index_dict)
    index2newsid_dict_test = invert_dict(newsid2index_dict_test)
    
    # 构建train矩阵
    train_data_matrix = create_matrix(train_data,userid2index_dict,newsid2index_dict)
    # svd分解
    (svd_nonread,svd_readed) = svd(train_data_matrix)
    
    for user in range(user_num):
        arr=svd_nonread[user]
        res=arr.argsort()[-old_num:][::-1]
        rec_bysvd[user]=res 
    rec_bysvd = rec_bysvd.astype(int) #转换过的newsid 旧新闻 
    # 热点新闻
    train_news_click = train_data_matrix.sum(axis=0)
    
    df_train_news_click = pd.DataFrame(data={'newsid':[index2newsid_dict[i] for i in range(news_num)],'click':list(train_news_click)})  # 原本的id
    df_train_news_click.sort_values(by=['click'], inplace=True, ascending=False)
    df_train_news_click.reset_index(inplace=True,drop=True)
    # df_train_news_click.to_csv('train_news_click.csv', index_label='index') # b保存
    
    
    # 计算每个test新闻里对应的train里新闻的相似度,
    sim_news_inverse=calsim(df_id2keywords_test, df_id2keywords,total_num)
    # 根据热点新闻和其相似新闻推荐 
    rec_freq = freqclicknews_rec(df_train_news_click,sim_news_inverse,newsid2index_dict,index2newsid_dict_test,total_num,old_num,new_num)
        
        
    # 推荐
    rec={}
    #真实的userid
    for user in test_data.userid.unique():
            if(user in train_data.userid.unique()):       
                predict = list(rec_bysvd[userid2index_dict[user]])# train里的userindex
                rec_new=[]
                for k in range(new_num):
                    df_rec_bysvd = pd.DataFrame([x for x in sim_news_inverse[predict[k]][:per_num]], columns=['newsid_test', 'similarity'])
                    df_rec_bysvd.sort_values(by=['similarity'], inplace=True)
                    df_rec_bysvd.reset_index(inplace=True,drop=True)
                    for j in range(per_num):
                        rec_new.append(index2newsid_dict_test[df_rec_bysvd['newsid_test'][j]])    # news_id

                for i in range(len(predict)):
                    predict[i] = index2newsid_dict[predict[i]]
                predict.extend(rec_new)
                rec[user] = predict
            else:

                rec[user] = rec_freq
    return rec
# 评价
def precision_k(df_rec,test_data,k):
    test_data_dict={}
    for userid,group in test_data.groupby('userid'):
        test_data_dict[userid] = group['newsid'].values.tolist()
    precision_k=[]
    for i in range(df_rec.shape[0]):#每个user
        count = 0
        read_list = df_rec['rec_news'][i]
        for j in read_list:# 遍历推荐list
            if j in test_data_dict[df_rec['userid'][i]]:
                count+=1
                if count == k:
                    break
        precision_k.append(count/(read_list.index(j)+1))

    return np.mean(precision_k)#,precision_k




def MAP(df_rec,test_data):
    test_data_dict={}
    for userid,group in test_data.groupby('userid'):
        test_data_dict[userid] = group['newsid'].values.tolist()
    MAP=[]
    for i in range(df_rec.shape[0]):#每个user
        count = 0
        sum=0
        read_list = df_rec['rec_news'][i]
        for j in read_list:# 遍历推荐list
            if j in test_data_dict[df_rec['userid'][i]]:
                count+=1
                sum+=count/(read_list.index(j)+1)
        MAP.append(sum)
    return np.mean(MAP)



def RR(df_rec,test_data):
    test_data_dict={}
    for userid,group in test_data.groupby('userid'):
        test_data_dict[userid] = group['newsid'].values.tolist()
    RR=[]
    for i in range(df_rec.shape[0]):#每个user
        #count = 0
        sum=0
        read_list = df_rec['rec_news'][i]
        for j in read_list:# 遍历推荐list
            if j in test_data_dict[df_rec['userid'][i]]:
                #count+=1
                sum+=1/(read_list.index(j)+1)
        RR.append(sum/len(read_list))
    return np.mean(RR)





if __name__ == "__main__":

    rec = recommend(train_data,test_data,newsid2index_dict,userid2index_dict,newsid2index_dict_test,10,8,1)

    user_list=[]
    rec_list=[]
    for key in rec:
        user_list.append(key)
        rec_list.append(rec[key])
    df_rec = pd.DataFrame({"userid":user_list,"rec_news":rec_list})

    df_rec.to_csv("recommendation_result.csv",index=False)
    # 10 8 1
    print(precision_k(df_rec,test_data,1))
    print(precision_k(df_rec,test_data,5))
    print(precision_k(df_rec,test_data,10))
    print(MAP(df_rec,test_data))
    print(RR(df_rec,test_data))

