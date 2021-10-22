# test file

from pandas.core import frame
import requests
import json
#from Elastic_Net_file import Elastic_net
#from sklearn.linear_model import ElasticNet,ElasticNetCV
import sklearn
from sklearn.metrics import mean_squared_error
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import close, times
from numpy.lib.financial import nper
import numpy
import tensorflow
import math
import pandas
import gym
import random
from Qlearning import DeepQlearning, Agent
import csv
import datetime
import calendar


alphas = []

for x in range(0,100):
    alphas.append.random_number(0.1,.8)
    


def CSV_weekday(x, y):

    weekday = []

    dates = []

    hour = []

    minute = []

    #CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice=year1month2&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    URL1 = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice='
    URL2 = '&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    CSV_URL = URL1 + 'year' + x + 'month' + y + URL2
    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        response = list(cr)

        z = 0
        for row in response:

            if(z > 0):
                date = row[0]

                dates.append(date)

            z = z+1


        #print(dates)
            




    data = []
    x = 0
    for i in dates:
#         date_data = x[0:10]
        year = i[0:4]
        month = i[5:7]
        day = i[8:10]
        x1 = int(year)
        x2 = int(month)
        x3 = int(day)
        dt = datetime.datetime(x1, x2, x3, 0, 0, 0, 0)
        #print(calendar.day_name[dt.weekday()])
        data.append(calendar.day_name[dt.weekday()])
        x = x+1
    #print(x)
    return data


def CSV_hour(x, y):
    
    weekday = []

    dates = []

    #CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice=year1month2&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    URL1 = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice='
    URL2 = '&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    CSV_URL = URL1 + 'year' + x + 'month' + y + URL2
    
    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        response = list(cr)

        z = 0
        for row in response:

            if(z > 0):
                date = row[0]

                dates.append(date)

            z = z+1


        #print(dates)
            




    data = []
    x = 0
    for i in dates:
#         date_data = x[0:10]
        year = i[0:4]
        month = i[5:7]
        day = i[8:10]
        hour = i[10:13]
        minute = i[13:15]
        x1 = int(year)
        x2 = int(month)
        x3 = int(day)
        x4 = int(hour)
        #x5 = int(minute)
        #print(calendar.day_name[dt.weekday()])
        data.append(x4)
        x = x+1
    #print(x)
    return data

def CSV_minute(x, y):
    
    weekday = []

    dates = []

    #CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice=year1month2&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    
    URL1 = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=AAPL&interval=1min&slice='
    URL2 = '&outputsize=full&apikey=ZRFYTH4TBX2BA01'
    CSV_URL = URL1 + 'year' + x + 'month' + y + URL2
    
    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        response = list(cr)

        z = 0
        for row in response:

            if(z > 0):
                date = row[0]

                dates.append(date)

            z = z+1


        #print(dates)
            




    data = []
    x = 0
    for i in dates:
#         date_data = x[0:10]
        year = i[0:4]
        month = i[5:7]
        day = i[8:10]
        hour = i[12:13]
        minute = i[14:16]
        x1 = int(year)
        x2 = int(month)
        x3 = int(day)
        x4 = int(hour)
        x5 = int(minute)
        #print(calendar.day_name[dt.weekday()])
        data.append(x5)
        x = x+1
    #print(x)
    return data
  
#Data Sets


volume = []

weekday = []

temp_weekday = []

hour = []

minute = []

time = []

for i in range(1,3)
    for x in range(1,13)
     temp_weekday = CSV_weekday(i,x)
     volume.append(CSV_volume(i,x))
     hour.append(CSV_hour(i,x))
     minute.append(CSV_minute(i,x))

for i in temp_weekday:
    for x in i
        if x == 'Monday'
            weekday.append(1)
        elif x == 'Tuesday'
           weekday.append(2)
        elif x == 'Wednesday'
            weekday.append(3)
        elif x == 'Thursday'
            weekday.append(4)
        elif x == 'Friday'
            weekday.append(5)
        elif x == 'Saturday'
           weekday.append(6)
        elif x == 'Sunday'
            weekday.append(7)
        else
            print('check temp_weekday if staments and loop')
            
        
        
        
        
        
        
        
 
        
#recheck if this works in elastic net 
#might need to change str and use the added value of intigers on the x axis of elastic net insted of time
        
        
        
for i in weekday:
    for x in hour:
     for y in minute:
        time.append(str(weekday[y]*10000000) + str(minute))
        
        
 



#normalize data for elastic net first before running 



        
 #Elastic_net(high, time, alphas))


#make first param 2 dimentional
        
 
#starting_data2 = Elastic_net(volume), time, alphas)
    
 
#Deep Q-Learning

x1 = starting_data1
x2 = x1[0:16500]

def preprocess_data(data):
    return data.reshape(data, (1650,10))

def stack_data(stacked_data, x, buffer):

    if stacked_data is None:
        stacked_data = numpy.zeros((buffer, *x.shape))
        for i, _ in enumerate(stacked_data):
            stacked_data[i,:] = x 
    else:
        stacked_data [0:buffer-1,: ] = stacked_data[1:, :]
        stacked_data[buffer-1,:] = x

    stacked_data = stacked_data.reshape(1, *frame.shape[0:2], buffer)

    return stacked_data


stacked_data = x2
    

if __name__ == '_main_':

    env = gym.make('test1')

    load_checkpoint = False

    agent1 = agent(gamma = .99, epsilon = 1.0, aplha = .00025, input_dims = (10000, 1, 10), n_actions = 2, mem_size = 80000, batch_size = 4125)

    if load_checkpoint:
        agent1.load_models()

    scores = []
    numGames = 1000
    stack_size = 1650
    score = 0


    while agent1.mem_cntr < 80000:
        done = False
        observation = env.reset()
        observation = preprocess_data(observation)
        stacked_data = None
        observation = stack_data(stacked_data, observation, stack_size)


        z = 0
        while not done:
            action = numpy.random.choice([0,.01,-.01])
            #actions += 1
            observation_, rewards, done, info = env.step(action)
            observation_ = stack_data(stacked_data, preprocess_data(observation_), stack_size)
            #action -= 1

            reward = stacked_data[z] - observation_
            
            # or reward = (stacked_data[z]- (2*stacked_data[z])) + observation
            

            agent1.store_transistion(observation, action, reward, observation_, int(done))

            observation = observation_

print('Done with 1')

obversvation = x2

for i in range(0,1000):
    numGames.append(i)

for i in numGames:
    done = False 
    if i%10 == 0 and i > 0:
        avg_score = numpy.mean(scores[max(0, i-10):(i+1)])
        print('test', i, 'score', score, 'average_score %.3f', avg_score, 'epsilon %.3f', agent1.epsilon)

        agent1.save_models()
    else:
        print('test: ', i, 'score', score)
    
    observation = env.reset()
    observation = preprocess_data(observation)
    stacked_data = None
    observation = stack_data(stacked_data, observation, stack_size)

    while not done:
            action = agent1.choose.action(observation)
            #actions += 1
            observation_, rewards, done, info = env.step(action)
            observation_ = stack_data(stacked_data, preprocess_data(observation_), stack_size)
            #action -= 1
            agent1.store_transistion(observation, action, reward, observation_, int(done))

            observation = observation_

            agent1.learn()

            scores += reward 

scores.append(score)

         
#store_transistion(self, state, action, reward, state_, terminal)
#Feature Scaling

time_normalized = []
volume_normalized = []

for x1 in get_time:
    time_normalized = get_time[x1]  

for x1 in get_volume:
    volume_normalized = get_volume[x1]



