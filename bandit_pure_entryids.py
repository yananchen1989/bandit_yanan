from __future__ import division
import sys
from datetime import datetime
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np 
from scipy.stats import beta
import math
import traceback
import memcache
import json
import redis
import pandas as pd 
#from sklearn.externals import joblib
#import pylru
from news_client import *
import sys
import pickle
import os
from  MemcacheHandler import *
infra_redis_host = '172.17.31.96'
infra_redis_port = 6666
#valid_news_prefix_list = ['DAY:N:C:L:', 'EVER:N:C:L:']
prefix = 'DAY:N:C:L:'
host = ['172.17.28.150','172.17.28.149', '172.17.28.143','172.17.28.179','172.17.28.183'] #172.17.28.142 
port = [7070 for _ in host]

#country = 'ng'#sys.argv[1]
language = 'en'#sys.argv[2]
sel_cates1 = ['News_Politics','Others_Terrorism','News_Sports','Crime','Policy','Religion','News_Entertainment',\
             'Accident','Science','Travel','Prose','Fashion&Beauty','Food','Motoring',\
             'Business','Education','News_Lifestyle','Health','Pet_Animals','Art&Design','Technology']
sel_cates0 =['News_Politics','News_Sports','News_Entertainment']

def get_valid_news():
    redis_client = redis.StrictRedis(host=infra_redis_host, port=infra_redis_port, db=0)
    valid_news = set() # store all news ids
    for country in ['ng','za','gh','tz','ke']:
        key = prefix + country + ":" + language
        res = redis_client.hkeys(key)
        valid_news.update(res)
        logging.info("[country:%s, language:%s] Totally get %d valid news.", country, language, len(res))
    logging.info("[country:%s, language:%s] Totally get %d valid news.", country, language, len(valid_news))
    return valid_news

def get_profile(entry_id_list):
    news_client = NewsClient(host, port)
    lack_entry_list = []
    res_dic = {}
    #news_profile_dic = {}

    for entry in entry_id_list:
        lack_entry_list.append(entry)
    logging.info("Lack %d news profile, request remote", len(lack_entry_list))
    lack_profile_dic = news_client.get_profile(lack_entry_list)
    for k, v in lack_profile_dic.items():
        if v:
            topic64  = v.get('topic64', '')
            topic256 = v.get('topic256','')
            category_v2_score = v.get('category_v2_score', '')
            cate = v.get('category', '')
            if topic64 == '' or topic256 == '' or category_v2_score == '' or cate == '':
                continue
            res_dic[k] = [cate, str(topic64), str(topic256), str(category_v2_score)]
            #news_profile_dic[k] = [topic64, topic256, category_v2_score]
    logging.info("Totally get %d news profile.", len(res_dic))
    return res_dic

def get_status(entry_id_list):
    client = NewsClient(host, port)
    res_status = client.get_status(entry_id_list)
    logging.info("Totally get %d news status.", len(res_status))
    return res_status

def thompson_sampling(ck, N_ta):
    reward = np.random.beta(ck, N_ta-ck)
    return reward
def parse_topic_to_list(dic, dim):      
    line = [0] * dim    
    for i in dic.keys():    
        line[int(i)] = dic[i]    
    return line
def parse_cate_to_list(dic, sel_cates ):     
    origin = {}    
    for c in sel_cates:    
        origin[c] = 0    
    for i in dic.keys():    
        origin[i] = dic[i]    
    line = []    
    for i in sel_cates:    
        line.append(origin[i])    
    return line 

while True:
    logging.info('-------------------------')
    entry_id_list = list(get_valid_news()) # list: '97784d95_tz', 'eddf972a_za', '1dd0b95f_ng'
    #entry_id_list = entry_id_list[:1000]
    news_profile = get_profile(entry_id_list) # dict 'ae113049_za': u'News_Sports',
    with open('news_profile_dic_', 'w') as f:
        pickle.dump(news_profile, f)

    logging.info('........news_profile_dic written to disk.......') 
    news_status = get_status(entry_id_list) # dict
    #break
    news_profile_keys = set(news_profile.keys())
    news_status_keys = set(news_status.keys())
    
    df = []
    lose_cnt = 0
    for id in entry_id_list:
        if id not in news_profile_keys or id not in news_status_keys:
            lose_cnt += 1
            continue 
        #if news_profile[id][0] not in sel_cates0  
        if news_status[id]['total_impressions'] <= 5 or news_status[id]['total_impressions'] >= 500:
            lose_cnt += 1
            continue
        if news_status[id]['total_impressions'] <=  news_status[id]['total_clicks']:
            lose_cnt += 1
            continue
        if news_status[id]['total_clicks'] == 0:
            lose_cnt += 1
            continue
        ts_score = thompson_sampling(news_status[id]['total_clicks'], news_status[id]['total_impressions']) 
        df.append([id, news_profile[id][0], news_status[id]['total_impressions'], news_status[id]['total_clicks'],\
                           ts_score, news_profile[id][1], news_profile[id][2], news_profile[id][3]])

    logging.info('lose_cnt: %d', lose_cnt )
    df = pd.DataFrame(df)
    df.columns=['entry_id','category','pv','ck', 'ts_score', 'topic64','topic256', 'category_v2_score'] 
    cut = 30000
    logging.info('df_cnt: %d', df.shape[0])
    #assert df.shape[0] > cut
    df_cut = df.sort_values(by=['ts_score'], ascending=False)[['entry_id','topic64','topic256', 'category_v2_score']].values.tolist()[:cut]
    
    df_topic_cate = []
    for i in df_cut:
        id = i[0]
        topic64_vec = parse_topic_to_list(eval(i[1]), 64)
        topic256_vec = parse_topic_to_list(eval(i[2]), 256)
        cates_vec = parse_cate_to_list(eval(i[3]), sel_cates1)
        if len(topic64_vec) != 64 or len(topic256_vec) != 256 or len(cates_vec) != 21:
            continue
        df_topic_cate.append([id, topic64_vec, topic256_vec, cates_vec])
    if len(df_topic_cate) != cut:
        logging.info("df_topic_cate length error, actual length: %d", len(df_topic_cate))
        continue
    logging.info("df_topic_cate cnt: %d ", len(df_topic_cate))
    with open('df_topic_cate_', 'w') as f:
        pickle.dump(df_topic_cate, f) # 21M
    logging.info('df_topic_cate written .......') 
    os.system("mv df_topic_cate_ df_topic_cate")
    os.system("mv news_profile_dic_ news_profile_dic")
    logging.info('................files renamed........................')
    logging.info('begin sleep........................')


