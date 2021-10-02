# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:15:42 2020

@author: Mahdi
"""

import numpy as np
# import threading
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.initializers import random_normal

# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

batch_size = 32

# A class for the DQN model
class DQNModel():
    def __init__(self, weights_path, train_conv_layers):
        self.__action_context = tf.compat.v1.get_default_graph() # Returns the default Graph being used in the current thread.
        # Graph is a TensorFlow computation, represented as a dataflow graph.
        # Graphs are used by tf.functions to represent the function's computations. Each graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.
        with self.__action_context.as_default():
            self.__angle_values = [-0.9, -0.5, 0, 0.5, 0.9]
    
            self.__nb_actions = 5
            self.__gamma = 0.99
    
            #Define the model
            activation = 'relu'
            pic_input = Input(shape=(59,255,3))
            
            img_stack = Conv2D(16, (3, 3), name='convolution0', padding='same', activation=activation, trainable=train_conv_layers)(pic_input)
            img_stack = MaxPooling2D(pool_size=(2,2))(img_stack)
            img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1', trainable=train_conv_layers)(img_stack)
            img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
            img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2', trainable=train_conv_layers)(img_stack)
            img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
            img_stack = Flatten()(img_stack)
            img_stack = Dropout(0.2)(img_stack)
    
            img_stack = Dense(128, name='rl_dense', kernel_initializer=random_normal(stddev=0.01))(img_stack)
            img_stack=Dropout(0.2)(img_stack)
            output = Dense(self.__nb_actions, name='rl_output', kernel_initializer=random_normal(stddev=0.01))(img_stack)
    
            opt = Adam()
            self.__action_model = Model(inputs=[pic_input], outputs=output)
    
            self.__action_model.compile(optimizer=opt, loss='mean_squared_error')
            self.__action_model.summary()
            
            # If we are using pretrained weights for the conv layers, load them.
            if (weights_path is not None and len(weights_path) > 0):
                print('Loading weights')
                print('Current working dir is {0}'.format(os.getcwd()))
                self.__action_model.load_weights(weights_path, by_name=True)
            else:
                print('Not loading weights')
    
            # Set up the target model. 
            # This is a trick that will allow the model to converge more rapidly.
            self.__target_model = clone_model(self.__action_model)
    
            self.__target_context = tf.compat.v1.get_default_graph()

    # A helper function to read in the model from a JSON packet.
    # This is used to read the file from disk
    def from_packet(self, packet):
        with self.__action_context.as_default():
            self.__action_model.set_weights([np.array(w) for w in packet['action_model']])
            self.__action_context = tf.compat.v1.get_default_graph()
            print('Received action Model from JSON')
        if 'target_model' in packet:
            with self.__target_context.as_default():
                self.__target_model.set_weights([np.array(w) for w in packet['target_model']])
                self.__target_context = tf.compat.v1.get_default_graph()
                print('Received target Model from JSON')

    # A helper function to write the model to a JSON packet.
    # This is used to send the model
    def to_packet(self, get_target = True):
        packet = {}
        with self.__action_context.as_default():
            packet['action_model'] = [w.tolist() for w in self.__action_model.get_weights()]
            self.__action_context = tf.compat.v1.get_default_graph()
        if get_target:
            with self.__target_context.as_default():
                packet['target_model'] = [w.tolist() for w in self.__target_model.get_weights()]

        return packet

            
    def update_critic(self):
        with self.__target_context.as_default():
            self.__target_model.set_weights([np.array(w, copy=True) for w in self.__action_model.get_weights()])
    
            
    # Given a set of training data, trains the model.
    # The agent will use this to compute the model updates
    def update_action_model_from_batches(self, batches):
        pre_states = np.array(batches['pre_states'])
        post_states = np.array(batches['post_states'])
        rewards = np.array(batches['rewards'])
        actions = list(batches['actions'])
        is_not_terminal = np.array(batches['is_not_terminal'])
        
        # For now, our model only takes a single image in as input. 
        # Only read in the last image from each set of examples
        pre_states = pre_states[:, 3, :, :, :]
        post_states = post_states[:, 3, :, :, :]
        
        print('START UPDATING ACTION MODEL')
        
        # We only have rewards for the actions that the agent actually took.
        # To prevent the model from training the other actions, figure out what the model currently predicts for each input.
        # Then, the gradients with respect to those outputs will always be zero.
        with self.__action_context.as_default():
            # q_lables is 2D Matrix with the shape of
            # [?][11] : [num of action in the episode (variant)][self.__nb_actions]
            q_lables = self.__action_model.predict([pre_states], batch_size=batch_size)
        
        # Find out what action the target model will predict for each post-decision state.
        with self.__target_context.as_default():
            q_futures = self.__target_model.predict([post_states], batch_size=batch_size)

        # Apply the Bellman equation
        # although we use epsilon-greedy policy to choose several actions
        # taken in episodes by predicting qs through action_model,
        # to update qs we always use the q_futures_max greedily
        # computed between episodes by predicting qs through target_model,
        # This is off-policy or behavioral policy
        # Q value = reward + discount factor * expected future reward
        q_futures_max = np.max(q_futures, axis=1)
        changed_q_lables = (q_futures_max * is_not_terminal * self.__gamma) + rewards
        
        # Update the label only for the actions that were actually taken.
        for i in range(0, len(actions), 1):
            # q_lables is 2D Matrix with the shape of
            # [?][5] : [num of action in the episode (variant)][self.__nb_actions]
            q_lables[i][actions[i]] = changed_q_lables[i]

        # Perform a training iteration.
        with self.__action_context.as_default():
            pre_weights = [np.array(w) for w in self.__action_model.get_weights()] # a list of array-type weights
            self.__action_model.fit([pre_states], q_lables, epochs=1, batch_size=batch_size, verbose=1)
            
            # Compute the change in weights
            new_weights = self.__action_model.get_weights()
            dx = 0
            for i in range(0, len(pre_weights), 1):
                dx += np.sum(np.sum(np.abs(new_weights[i]-pre_weights[i])))
            print('change in weights from training iteration: {0}'.format(dx))
        
        print('END UPDATING ACTION MODEL')


    # Performs a state prediction given the model input
    def predict_action_and_reward(self, observation):
        if (type(observation) == type([])):
            observation = np.array(observation)
        
        # Our model only predicts on a single state.
        # Take the latest image
        observation = observation[3, :, :, :]
        observation = observation.reshape(1, 59,255,3)
        with self.__action_context.as_default():
            predicted_qs = self.__action_model.predict([observation])
        # predicted_qs is 2D Matrix with the shape of
        # [?][5] : [num of action in the episode (variant)][self.__nb_actions]
        # return the action with the highest Q value "in the future" -> np.argmax
        # and its exact next q value as reward -> [0]
        predicted_action = np.argmax(predicted_qs)
        return (predicted_action, predicted_qs[0][predicted_action])

    # Convert the current state to control signals to drive the car.
    # As we are only predicting steering angle, we will use a simple controller to keep the car at a constant speed
    def state_to_control_signals(self, action, car_state):
        if car_state.speed < 1:
            return (self.__angle_values[action], 0.6, 0)
        elif car_state.speed < 3:
            return (self.__angle_values[action], 0.5, 0)
        elif car_state.speed < 6:
            return (self.__angle_values[action], 0.2, 0)
        else :
            return (self.__angle_values[action], 0, 1)

    # Gets a random action
    # Used during annealing
    def get_random_action(self):
        return np.random.randint(low=0, high=self.__nb_actions)
