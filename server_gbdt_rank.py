# -*- coding:utf-8 -*-
import lightgbm as lgb
import os 
import time
import redis
import random
import logging
import threading
import numpy as np
from flask import Flask
from flask import request
import itertools
from flask import make_response
from news_client import *
import pickle
app = Flask(__name__)
sel_cates = ['News_Politics','Others_Terrorism','News_Sports','Crime','Policy','Religion','News_Entertainment',\
                'Accident','Science','Travel','Prose','Fashion&Beauty','Food','Motoring',\
                'Business','Education','News_Lifestyle','Health','Pet_Animals','Art&Design','Technology']
nations = ["gh", "ke", "ng", "tz", "za"]
#logging.basicConfig(filename = './server_gbdt_rank.log')
redis_client = redis.StrictRedis(host="172.17.29.71", port=6666, db=0)
#lgb_model = lgb.Booster(model_file='lgb_model')
logging.info("lgb_model loaded")
'''
host = ['172.17.28.150','172.17.28.149', '172.17.28.143','172.17.28.179','172.17.28.183'] #172.17.28.142
port = [7070 for _ in host]
'''
class TSNew:
    def __init__(self):
        self.thread = threading.Thread(target=self.run)
        self.ts_picked_ids = None
        self.thread.start()

    def run(self):
        while True:            
            self.lgb_model = lgb.Booster(model_file='lgb_model')
            with open('./df_topic_cate', 'r') as f :
                df_topic_cate_ = pickle.load(f)
            df_topic_cate_nationdic = {}
            entry_ids_nationdic = {}
            feats_nationdic = {}
            for n in nations:
                df_topic_cate_nationdic[n] = []
                entry_ids_nationdic[n] = []
                feats_nationdic[n] = []
            for i in df_topic_cate_:
                n = i[0].split('_')[1]
                df_topic_cate_nationdic[n].append(i)
                entry_ids_nationdic[n].append(i[0])
                feats_nationdic[n].append(i[1] + i[2] + i[3])
            for n in nations:
                feats_nationdic[n] = np.array(feats_nationdic[n])
            self.df_topic_cate = df_topic_cate_nationdic
            self.entry_ids = entry_ids_nationdic
            self.feats = feats_nationdic
            logging.info("~~~~~~~~~~~df_topic_cate loaded ~~~~~~~~~~~~")

            with open('./news_profile_dic', 'r') as f:
                news_profile_dic_ = pickle.load(f)
            self.news_profile_dic = news_profile_dic_
            logging.info("~~~~~~~~~~~news_profile_dic loaded ~~~~~~~~~~~~")
            logging.info("~~~~~~~~~~~dic loaded success ~~~~~~~~~~~~")
            time.sleep(60*20)
TSNew_ = TSNew()

def get_mean(m):
    return [float(sum(l))/len(l) for l in zip(*m)]

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
'''
def get_user_profile_from_history_news(user_history_entry_ids):
    news_client = NewsClient(host, port)
    profile_dic = news_client.get_profile(user_history_entry_ids)
    line = []
    for k,v in profile_dic.items():
        if v:
            topic64 = v.get('topic64','')
            topic256 = v.get('topic256','')
            category_v2_score = v.get('category_v2_score', '')
            if topic64 == '' or topic256 == '' or category_v2_score == '':
                continue
            topic64_vec = parse_topic_to_list(topic64, 64)
            topic256_vec = parse_topic_to_list(topic256, 256)
            cates_vec = parse_cate_to_list(category_v2_score, sel_cates)
            concate = topic64_vec + topic256_vec + cates_vec 
            line.append(concate)
    if line == []:
        return []
    return get_mean(line) 
'''
def get_user_profile_from_history_news_from_dict(user_history_entry_ids, news_profile_dic):
    line = []
    for id in user_history_entry_ids:
        value = news_profile_dic.get(id,'')
        if value == '':
            continue
        topic64 = eval(value[1])
        topic256 = eval(value[2])
        category_v2_score = eval(value[3])
        topic64_vec = parse_topic_to_list(topic64, 64)
        topic256_vec = parse_topic_to_list(topic256, 256)
        cates_vec = parse_cate_to_list(category_v2_score, sel_cates)
        concate = topic64_vec + topic256_vec + cates_vec 
        line.append(concate)
    if line == []:
        return []
    return get_mean(line) 

def backup_news(df_topic_cate):
    backup_response =  [(i[0], 1) for i in df_topic_cate[:500]]
    return str(backup_response)
def get_user_result():
    logging.info("----------------come in ------------------")
    if request.method == 'POST':
            
        infom = request.get_data()# userid
        print infom
        if infom == '':
            logging.info("----------------infom is ''!!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------")
            return ''

        tokens = infom.split('`')
        if len(tokens) != 5:
            logging.info("----------------infom tokens cnt is not 5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------")
            return ''
        userid = tokens[0]
        cut = int(tokens[1])
        final_cut = int(tokens[2])
        history_cnt = int(tokens[3])
        country = tokens[-1]

        df_topic_cate = TSNew_.df_topic_cate
        feats = TSNew_.feats
        entry_ids = TSNew_.entry_ids
        news_profile_dic = TSNew_.news_profile_dic
        lgb_model = TSNew_.lgb_model
            
        logging.info("----------------userid %s ----------------", userid )
        logging.info("---country: %s", country)
        #backup_response = backup_news(df_topic_cate[country])
        #userid = 'ffe541a245604d46ecb9fd2d3149370c6bc2c92a'
        user_history = redis_client.zrevrange('uclick:%s' % userid , 0,100, withscores=True) # 255 µs
        logging.info("user_history returned news cnt: %d", len(user_history) )
        # user history: [('60980663_ng', 1524262272.0), ('951d00a2_ng', 1524262240.0), ('90f00a37_ng', 1524262151.0)]
        if len(user_history) == 0:
            logging.info("-------------user_history news zero--------------\n")
            return backup_news(df_topic_cate[country])

        user_history_entry_ids = [i[0] for i in user_history[:history_cnt]]      
        logging.info("%d user history entry_ids in fact ", len(user_history_entry_ids))
        user_vec = get_user_profile_from_history_news_from_dict(user_history_entry_ids, news_profile_dic) 
        if user_vec == [] or len(user_vec) != (64+256+21): # 341
            logging.info("user_vec real length: %d ", len(user_vec))
            logging.info("----------can't find feats for the user's history news------userid: %s  \n", userid)
            return backup_news(df_topic_cate[country])
        if feats[country].shape[0] < cut:
            cut = feats[country].shape[0]
            logging.info("shrink cut: %d", cut) 
        feats_news_user = np.concatenate((feats[country][:cut],  np.tile(user_vec,(cut,1)) ), axis=1)
        pred_scores = lgb_model.predict(feats_news_user)# 1000: 8ms
        if len(pred_scores) != cut:
            logging.info("GBDT pred_scores error:  %d", len(pred_scores))
            return backup_news(df_topic_cate[country])
        result = zip(entry_ids[country][:cut], list(pred_scores)) # 73.8 µs
        result_sort = sorted(result, key=lambda l:l[1], reverse=True) # 454 µs
        response = make_response(str(result_sort[:final_cut])) # param
        logging.info("----------response success ---------------\n")

        return response # "1,2,3,4,5"

@app.route('/', methods=['POST'])
def get_ts_gbdt_id():
    #tool_index = random.sample(range(10),1)[0]
    return get_user_result()

from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3333)


