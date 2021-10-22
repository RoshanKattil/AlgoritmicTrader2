# test file
from functools import lru_cache
import os
import numpy as numpy
from numpy import random
from numpy.core.fromnumeric import choose, cumprod, shape, size
from sklearn.base import TransformerMixin
import tensorflow as tensorflow
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_data_flow_ops import sparse_accumulator_apply_gradient
from tensorflow.python.ops.gen_nn_ops import conv2d
from tensorflow.python.ops.math_ops import reduce_mean

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

from functools import lru_cache
import os
import numpy as numpy
from numpy import random
from numpy.core.fromnumeric import choose, cumprod, shape, size
from sklearn.base import TransformerMixin
import tensorflow as tensorflow
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_data_flow_ops import sparse_accumulator_apply_gradient
from tensorflow.python.ops.gen_nn_ops import conv2d
from tensorflow.python.ops.math_ops import reduce_mean



#adjust input dims from 500 to max values of what data is being used
class DeepQlearning(object):

    def _init_(self, lr, n_actions, name , fcl_dims= 10000, input_dims=(10000,4125), chkpt_dir = 'E:'):

            self.lr = lr
            self.name = name
            self.n_actions = n_actions
            self.fcl_dims = fcl_dims
            self.input_dims = fcl_dims
            self.sees = tensorflow.Session()
            self.build_network()
            self.sess.run(tensorflow.global_variables_intializer())
            self.tensorflow.train.Saver()
            self.checkpint_file = os.path.join(chkpt_dir, 'deepnet.ckpt')
            self.params = tensorflow.get_colection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_net(self):
        with tensorflow.variable_scope(self.name):
            self.input = tensorflow.placeholder(tensorflow.float32, shape=[None, *self.input_dims], nome = 'inputs')
            self.acitons = tensorflow.placeholder(tensorflow.float32, shape = [None, self.n_actions])



            #intialize with elastic net 

            conv1 = tensorflow.layers.features = tensorflow.layers.conv2d(inputs = self.input, filters=32, kernel_size= (8,8), strides = 4, name="conv1", 
            

            kernal_instializer = tensorflow.variance_scaling_intializer(scale = 2))




            #change later



            conv1_activated = tensorflow.nn.relu(conv1)


            #intialize with elastic net 

            conv2 = tensorflow.layers.features = tensorflow.layers.conv2d(inputs = self.input, filters=64, kernel_size= (4,4), strides = 2, name="conv2", 
            

            kernal_instializer = tensorflow.variance_scaling_intializer(scale = 2))




            #change later

            conv2_activated = tensorflow.nn.relu(conv2)
            

            #intialize with elastic net 

            conv3 = tensorflow.layers.features = tensorflow.layers.conv2d(inputs = self.input, filters=128, kernel_size= (3,3), strides = 1, name="conv3", 
            

            kernal_instializer = tensorflow.variance_scaling_intializer(scale = 2))




            #change later

            conv3_activated = tensorflow.nn.relu(conv3)

            flat = tensorflow.layers.flatten(conv3_activated)

            densel = tensorflow.layers.dense(flat, units=self.fcl_dims, activation = tensorflow.nn.relu, kernal = tensorflow.variance_scaling_intializer(scale = 2))

            self.Q_values = tensorflow.layers.dense(densel, units = self.n_actions, kernal_initalizer = tensorflow.varience_scaling_instalizer(scale=2))

            self.q = tensorflow.reduce_sum(tensorflow.multiply(self.Q_values, self.acitons))

            self.loss = tensorflow(reduce_mean(tensorflow.square(self.q - self.q_target)))

            self.train_op = tensorflow.train.AdamOptimizer(self.lr.minimise(self.loss))

    def Load_Checkpoint(self):
        print('loading')
        self.saver.restore(self.sees, self.checkpoint_file)

    def Save_Checkpoint(self):
        print('saving')
        self.saver.save(self.sees, self.checkpoint_file)
        
        
        
        
        
 #adjust 500 to what im training max size

class agent(object):
    def  _init_(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size, replace_target = 5000, input_dims = (10000, 4125,10), q_next= 'tmp/q_next', q_eval = 'tmp/q_eval'):

        self.n_actions = n_actions

        self.action_space = [i for i in range(self.n_actions)]

        self.gamma = gamma

        self.mem_size = mem_size

        self.epsilon = epsilon

        self.batch_size = batch_size

        self.cntr = 0

        self.replace_target = replace_target

        self.q_next = DeepQlearning(alpha, n_actions, input_dims = input_dims, name = 'q_next', chkpt_dir = 'tmp/q_next')

        self.q_eval = DeepQlearning(alpha, n_actions, input_dims=input_dims, name = 'q_eval', chckpt_dir = 'tmp/q_eval')

        self.state_memory = numpy.zeros((self.mem_size, *input_dims))

        self.new_state_memory = numpy.zeros((self.mem_size, self.n_actions))

        self.action_memory = numpy.zeros((self.mem_size), self.n_actions)

        self.reward_memory = numpy.zeros(self.mem_size, dtype = numpy.int8)

        self.terminal_memory = numpy.zeros(self.mem_size, dtype=numpy.in8)

    def store_transistion(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = numpy.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def chose_action(self, state):
        rand = numpy.random.random()
        if rand < self.epsilon:
            action = numpy.random.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values, feed_dict = {self.input: state})

            action = numpy.argmax(actions)
        return action

    def learning(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()


        max_mem = self.mem_cntr if self.mem_cetr < self.mem_size else self.mem_size 
            
        batch = numpy.random.choice(max_mem, self.batch_size)

        state_batch = self.state_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        action_batch = self.action_memory[batch]

        action_values = numpy.array([0,1,2], dtype = numpy.int8)
        action_indices = numpy.dot(action_batch, action_values)

        reward_batch = self.reward_memory[batch]

        terminal_batch = self.terminal_memory[batch]

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,feed_dict={self.q_eval.input: state_batch})

        q_next = self.q_next.sess.run(self.q_next.Q_values,feed_dict={self.q_next.input: new_state_batch})

        q_target = q_eval.copy()

        q_target[:, action_indices] = reward_batch + self.gamma*numpy.max(q_next, axis = 1)*terminal_batch

        _ = self.q_eval.sess.run(self.q_eval.train_op, feed_dict = {self.q_eval.input: state_batch, self.q_eval.actions: action_batch, self.q_eval.q_target: q_target})

        if self.mem_cntr > 1000000:
            if self.epsilon > .01:
                self.epsilon *= .9999999
            elif self.epsilon <=.01:
                self.epsilon = .01

    def save_models(self):
        self.q_eval_save_checkpoint()
        self.q_next.save_checkpoint()


    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t,e in zip(t_params, e_params):
            self.q_eval.sess.run(tensorflow.assign(t,e))
            
            
            
           
#add render function with matplotlib.pyplot
def render(data):
    for i in data:
        x = i[1]
        y = i[2]
        
        plt.scatter(x, y, s = area, alpha=.5)
        plt.show()

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



starting_data = []

    for i in range(0,16501):
        starting_data.append(time[i], volume[i])
 
#Deep Q-Learning

x1 = starting_data
x2 = x1[0:16500]

def preprocess_data(data):
    return data.reshape(data, (4125,10))

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
    
    
    
    
    
    #use custom env
    
    

    env = gym.make('test1')
    
    
    
    

    load_checkpoint = False

    #agent1 = agent(gamma = .99, epsilon = 1.0, aplha = .00025, input_dims = (10000, 1, 10), n_actions = 2, mem_size = 80000, batch_size = 4125)
    agent1 = agent(gamma = .99, epsilon = 1.0, aplha = .00025, input_dims = starting_data, n_actions = 2, mem_size = 80000, batch_size = 4125)

    
    
    if load_checkpoint:
        agent1.load_models()

    scores = []
    numGames = 10000
    stack_size = 4125
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

for i in range(0,10000):
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
#may or may not use 


time_normalized = []
volume_normalized = []

for x1 in get_time:
    time_normalized = get_time[x1]  

for x1 in get_volume:
    volume_normalized = get_volume[x1]
