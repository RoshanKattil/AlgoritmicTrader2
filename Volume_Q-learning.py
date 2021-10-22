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

    def _init_(self, lr, n_actions, name , fcl_dims= 10000, input_dims=(500,4125), chkpt_dir = 'E:'):

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
    def  _init_(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size, replace_target = 5000, input_dims = (500, 4125,10), q_next= 'tmp/q_next', q_eval = 'tmp/q_eval'):

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
