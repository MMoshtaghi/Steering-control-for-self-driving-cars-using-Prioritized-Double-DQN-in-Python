# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:11:46 2020

@author: Mahdi
"""

from airsim_client import *
from DQN_model import DQNModel
import time
import numpy as np
import json
import os
import datetime
import sys
import copy

import pandas as pd


# A class that represents the agent that will drive the vehicle, train the model.
class Agent():
    def __init__(self, parameters):
        required_parameters = ['data_dir', 'max_episode_runtime_sec', 'replay_memory_size', 'batch_size', 'min_epsilon', 'per_iter_epsilon_reduction', 'train_conv_layers']
        for required_parameter in required_parameters:
            if required_parameter not in parameters:
                print(required_parameter)
                raise ValueError('Missing required parameter {0}'.format(required_parameter))

        
        print('Starting time: {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)
        self.__model_buffer = None
        self.__model = None
        self.__airsim_started = False
        self.__data_dir = parameters['data_dir']
        self.__per_iter_epsilon_reduction = float(parameters['per_iter_epsilon_reduction'])
        self.__min_epsilon = float(parameters['min_epsilon'])
        self.__max_episode_runtime_sec = float(parameters['max_episode_runtime_sec'])
        self.__replay_memory_size = int(parameters['replay_memory_size'])
        self.__batch_size = int(parameters['batch_size'])
        self.train_from_json = bool((parameters['train_from_json'].lower().strip() == 'true'))
        self.__train_conv_layers = bool((parameters['train_conv_layers'].lower().strip() == 'true'))
        self.__epsilon = 0.05 if self.train_from_json else 1
        self.__num_batches_run = 0
        self.__num_episodes = 0
        self.__last_checkpoint_batch_count = 0
        
        if 'batch_update_frequency' in parameters:
            self.__batch_update_frequency = int(parameters['batch_update_frequency'])
        
        if 'weights_path' in parameters:
            self.__weights_path = parameters['weights_path']
        else:
            self.__weights_path = None

        self.__car_client = None
        self.__car_controls = None

        self.__experiences = {}
        self.__experiences['pre_states'] = []
        self.__experiences['post_states'] = []
        self.__experiences['actions'] = []
        self.__experiences['rewards'] = []
        self.__experiences['predicted_rewards'] = []
        self.__experiences['is_not_terminal'] = []

        self.__init_road_points()
        self.__init_reward_points()
        
        
    # Starts the agent
    def start(self):
        self.__run_function()

    # The function that will be run during training.
    # start AirSim, and continuously run training iterations.
    def __run_function(self):
        print('Starting run function')
        if self.train_from_json :
            MODEL_FILENAME = None # 'D:/Documents/spyder/DQNProject/DataDir/checkpoint/run/m3000_far15_act5_0_0.5_0.9 (5).json' #Your model goes here
            # load the model from disk
            print('Receiving Model from JSON')
            self.__model = DQNModel(None, self.__train_conv_layers)
            with open(MODEL_FILENAME, 'r') as f:
                checkpoint_data = json.loads(f.read())
                self.__model.from_packet(checkpoint_data['model'])

        else:
            self.__model = DQNModel(self.__weights_path, self.__train_conv_layers)
            
        # Connect to the AirSim exe
        self.__connect_to_airsim()

        # Fill the replay memory by driving randomly.
        print('Filling replay memory...')
        while True:
            print('Running Airsim episode.')
            try:
                if self.train_from_json :
                    self.__run_airsim_episode(False)
                else:
                    self.__run_airsim_episode(True)
                percent_full = 100.0 * len(self.__experiences['actions'])/self.__replay_memory_size
                print('Replay memory now contains {0} members. ({1}% full)'.format(len(self.__experiences['actions']), percent_full))

                if (percent_full >= 100.0):
                    break
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim while fillling replay memory. Attempting to reconnect.')
                self.__connect_to_airsim()
            
        # Get the latest model. Other agents may have finished before us.
        print('Replay memory filled. Starting main loop...')
        
        while True:
            try:
                if (self.__model is not None):

                    #Generate a series of training examples by driving the vehicle in AirSim
                    print('Running Airsim episode.')
                    self.__num_episodes += 1
                    experiences, action_count = self.__run_airsim_episode(False)

                    # If we didn't immediately crash, train on the gathered experiences
                    if (action_count > 0):
                        print('Generating {0} minibatches...'.format(action_count))

                        print('Sampling Experiences.')
                        # Sample experiences from the replay memory
                        sampled_experiences = self.__sample_experiences(experiences, action_count, False)

                        self.__num_batches_run += action_count
                        
                        # If we successfully sampled, train on the collected minibatches.
                        if (len(sampled_experiences) > 0):
                            print('Publishing AirSim episode.')
                            self.__publish_batch_and_update_model(sampled_experiences, action_count)
    
            # Occasionally, the AirSim exe will stop working.
            # For example, if a user connects to the node to visualize progress.
            # In that case, attempt to reconnect.
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim. Attempting to reconnect.')
                self.__connect_to_airsim()

    # Connects to the AirSim Exe.
    # Assume that it is already running. After 10 successive attempts, attempt to restart the executable.
    def __connect_to_airsim(self):
        attempt_count = 0
        while True:
            try:
                print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count))
                self.__car_client = CarClient()
                self.__car_client.confirmConnection()
                self.__car_client.enableApiControl(True)
                self.__car_controls = CarControls()
                print('Connected!')
                return
            except:
                print('Failed to connect.')


    # Appends a sample to a ring buffer.
    # If the appended example takes the size of the buffer over buffer_size, the example at the front will be removed.
    def __append_to_ring_buffer(self, item, buffer, buffer_size):
        if (len(buffer) >= buffer_size):
            buffer = buffer[1:]
        buffer.append(item)
        return buffer
    
    
    # Runs an interation of data generation from AirSim.
    # Data will be saved in the replay memory.
    def __run_airsim_episode(self, always_random):
        print('Running AirSim episode.')
        
        # Pick a random starting point on the roads
        starting_points, starting_direction = self.__get_next_starting_point()
        
        # Initialize the state buffer.
        # For now, save 4 images at 0.01 second intervals.
        state_buffer_len = 4
        state_buffer = []
        wait_delta_sec = 0.03

        print('Getting Pose')
        self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        # Currently, simSetPose does not allow us to set the velocity. 
        # So, if we crash and call simSetPose, the car will be still moving at its previous velocity.
        # We need the car to stop moving, so push the brake and wait for a few seconds.
        print('Waiting for momentum to die')
        self.__car_controls.steering = 0
        self.__car_controls.throttle = 0
        self.__car_controls.brake = 1
        self.__car_client.setCarControls(self.__car_controls)
        time.sleep(3)
        
        print('Resetting')
        self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        #Start the car rolling so it doesn't get stuck
        print('Running car for a few seconds...')
        self.__car_controls.steering = 0
        self.__car_controls.throttle = 0.5
        self.__car_controls.brake = 0
        self.__car_client.setCarControls(self.__car_controls)
        
        # While the car is rolling, start initializing the state buffer
        stop_run_time =datetime.datetime.now() + datetime.timedelta(seconds=2)
        while(datetime.datetime.now() < stop_run_time):
            time.sleep(wait_delta_sec)
            state_buffer = self.__append_to_ring_buffer(self.__get_image(), state_buffer, state_buffer_len)
        done = False
        actions = []
        rewards = []
        episode_dur = 0
        car_state = self.__car_client.getCarState()

        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_episode_runtime_sec)
        
        num_random = 0
        far_off = False
        rst = False
        
        # Main data collection loop
        while not done:
            collision_info = self.__car_client.getCollisionInfo()
            utc_now = datetime.datetime.utcnow()
            
            # Check for terminal conditions:
            # 1) Car has collided
            # 2) Car is stopped
            # 3) The run has been running for longer than max_episode_runtime_sec. 
            #       This constraint is so the model doesn't end up having to process huge chunks of data, slowing down training
            # 4) The car has run off the road
            if (collision_info.has_collided or car_state.speed < 1 or utc_now > end_time or rst):
                print('Start time: {0}, end time: {1}'.format(start_time, utc_now), file=sys.stderr)
                if (utc_now > end_time):
                    print('timed out.')
                    print('Full autonomous run finished at {0}'.format(utc_now), file=sys.stderr)
                done = True
                episode_dur = utc_now - start_time
                sys.stderr.flush() # "flush" the buffer to the terminal
            else:

                # The Agent should occasionally pick random action instead of best action
                do_greedy = np.random.random_sample()
                # pre_state should have the last 4 images in a list format
                pre_state = copy.deepcopy(state_buffer) # copy recursively
                if (do_greedy < self.__epsilon or always_random):
                    num_random += 1
                    action_of_pre_state = self.__model.get_random_action()
                    predicted_reward = 0
                    print('Model randomly pick action {0}'.format(action_of_pre_state))
                    
                else:
                    action_of_pre_state, predicted_reward = self.__model.predict_action_and_reward(pre_state)
                    print('Model predicts action {0}'.format(action_of_pre_state))

                # Convert the selected state to a control signal
                next_control_signals = self.__model.state_to_control_signals(action_of_pre_state, self.__car_client.getCarState())

                # Take the action
                self.__car_controls.steering = next_control_signals[0]
                self.__car_controls.throttle = next_control_signals[1]
                self.__car_controls.brake = next_control_signals[2]
                self.__car_client.setCarControls(self.__car_controls)
                
                # Wait for a short period of time to see outcome
                time.sleep(wait_delta_sec)

                # Observe outcome and compute reward from action
                state_buffer = self.__append_to_ring_buffer(self.__get_image(), state_buffer, state_buffer_len)
                car_state = self.__car_client.getCarState()
                collision_info = self.__car_client.getCollisionInfo()
                
                # Procedure to compute distance reward
                #Get the car position
                position_key = bytes('position', encoding='utf8')
                x_val_key = bytes('x_val', encoding='utf8')
                y_val_key = bytes('y_val', encoding='utf8')
                car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
                # Distance component is exponential distance to nearest line
                distance = 999
                #Compute the distance to the nearest center line
                for line in self.__reward_points: # e.g: (-251.21722656  -209.60329102  0.0    ,    -132.21722656  -209.60329102  0.0 )
                    local_distance = 0
                    length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2) # e.g: (X1-X2)^2 + (Y1-Y2)^2 = 14161.0
                    if (length_squared != 0):
                        # calc the projected point of the car_point on the road center line 
                        t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared)) # e.g: np.dot( (car_point - (X1,Y1,Z1)) , [119.   0.   0.]) / 14161.0
                        proj = line[0] + (t * (line[1]-line[0])) # if t=0 -> proj=(X1,Y1)  if t=1 -> proj=(X2,Y2)  proj = (X1,Y1) + t*[119.   0.   0.]
                        # calc the car_point distance from its projection on the road center line
                        local_distance = np.linalg.norm(proj - car_point)
                    distance = min(local_distance, distance)
                distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
                far_off = distance > THRESH_DIST_far
                rst = distance > THRESH_DIST_rst
                reward = self.__compute_reward(collision_info, car_state, action_of_pre_state, distance_reward, far_off)
                
                # Add the experience to the set of examples from this iteration
                # Add the list of each kind of data from this iteration to the replay memory
                # self.__add_to_replay_memory('pre_states') = [iter1,iter2,iter3,...]
                self.__add_to_replay_memory('pre_states', pre_state)
                self.__add_to_replay_memory('post_states', state_buffer)
                actions.append(action_of_pre_state)
                self.__add_to_replay_memory('actions', action_of_pre_state)
                rewards.append(reward)
                self.__add_to_replay_memory('rewards', reward)
                self.__add_to_replay_memory('predicted_rewards', predicted_reward)
                self.__add_to_replay_memory('is_not_terminal', 1)

        # Only the last state is a terminal state.
        self.__experiences['is_not_terminal'][-1] = 0
        # is_not_terminal = [1 for i in range(0, len(actions)-1, 1)]
        # is_not_terminal.append(0)
        # self.__add_to_replay_memory('is_not_terminal', is_not_terminal)

        percent_random_actiions = num_random / max(1, len(actions))
        print('Percent random actions: {0}'.format(percent_random_actiions))
        print('Num total actions: {0}'.format(len(actions)))
        
        # If we are in the main loop, reduce the epsilon parameter so that the model will be called more often
        if not always_random:
            self.__epsilon -= self.__per_iter_epsilon_reduction
            self.__epsilon = max(self.__epsilon, self.__min_epsilon)
            # Use this line to update the log file:
            sum_reward = sum(rewards)
            avg_reward = sum_reward/len(rewards)
            
            # load DataFrame
            d = {'episode number' : [self.__num_episodes], 'reward sum' : [sum_reward], 'reward avg' : [avg_reward], 'number of actions' : [len(actions)], 'episode duration' : [episode_dur], 'percent random actiions' : [percent_random_actiions]}
            df2 = pd.DataFrame( data = d )
            df2.set_index( 'episode number', inplace = True )
            df = pd.read_csv( 'log.csv', index_col = 0 )
            df = df.append( df2 )
            # export DataFrame to csv
            df.to_csv( 'log.csv' )
            # self.__model.tensorboard.update_stats(reward_avg=avg_reward, reward_sum=sum_reward, num_action=len(actions), percent_rnd_action=percent_random_actiions, epsilon=self.__epsilon, episode_duration=episode_dur)

            
        
        
        return self.__experiences, len(actions)

    # Adds a set of examples to the replay memory
    def __add_to_replay_memory(self, field_name, data):
        # To prevent memory issues :
        if (len(self.__experiences[field_name]) >= self.__replay_memory_size):
            self.__experiences[field_name] = self.__experiences[field_name][1:]
        self.__experiences[field_name].append(data)

    # Sample experiences from the replay memory
    def __sample_experiences(self, experiences, action_count, sample_randomly):
        sampled_experiences = {}
        sampled_experiences['pre_states'] = []
        sampled_experiences['post_states'] = []
        sampled_experiences['actions'] = []
        sampled_experiences['rewards'] = []
        sampled_experiences['predicted_rewards'] = []
        sampled_experiences['is_not_terminal'] = []

        # Compute the surprise factor, which is the difference between the predicted an the actual Q value for each state.
        # We can use that to weight examples so that we are more likely to train on examples that the model got wrong.
        suprise_factor = np.abs(np.array(experiences['rewards'], dtype=np.dtype(float)) - np.array(experiences['predicted_rewards'], dtype=np.dtype(float)))
        suprise_factor_normalizer = np.sum(suprise_factor)
        suprise_factor /= float(suprise_factor_normalizer) # the highest has the most chance to be sampled more due to p

        # Generate one minibatch for each frame of the run
        # np.shape that returns a tuple with each index having the number of corresponding elements so num of actions in a episode = suprise_factor.shape[0].
        for _ in range(0, action_count, 1):
            if sample_randomly:
                #set of sample indices 
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False))
            else:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), size=(self.__batch_size), replace=False, p=suprise_factor))
        
            sampled_experiences['pre_states'] += [experiences['pre_states'][i] for i in idx_set]
            sampled_experiences['post_states'] += [experiences['post_states'][i] for i in idx_set]
            sampled_experiences['actions'] += [experiences['actions'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            
        return sampled_experiences
        
     
    # Train the model on minibatches.
    def __publish_batch_and_update_model(self, batches, batches_count):
        # Train and get the gradients
        print('Publishing episode data')
        self.__model.update_action_model_from_batches(batches)

        # How often to update the target network
        
        if (self.__num_batches_run > self.__batch_update_frequency + self.__last_checkpoint_batch_count):
            self.__model.update_critic()
            
            checkpoint = {}
            checkpoint['model'] = self.__model.to_packet(get_target=True)
            checkpoint['batch_count'] = batches_count
            checkpoint_str = json.dumps(checkpoint) # converts into a json string.

            checkpoint_dir = os.path.join(os.path.join(self.__data_dir, 'checkpoint'), self.__experiment_name)
            
            if not os.path.isdir(checkpoint_dir):
                try:
                    os.makedirs(checkpoint_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                        
            file_name = os.path.join(checkpoint_dir,'{0}.json'.format(self.__num_batches_run)) 
            with open(file_name, 'w') as f:
                print('Checkpointing to {0}'.format(file_name))
                f.write(checkpoint_str)
            
            self.__last_checkpoint_batch_count = self.__num_batches_run
            

    # Gets an image from AirSim
    def __get_image(self):
        image_response = self.__car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
        image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
        # we only need the RGB and road area of the image in an array and float format so ... (model inpute shape = (59,255,3))
        return image_rgba[76:135,0:255,0:3].astype(float)

    # Computes the reward functinon based on the car position.
    def __compute_reward(self, collision_info, car_state, action_of_pre_state, distance_reward, far_off):
        # If the car has collided, the reward is always Negetive
        if (collision_info.has_collided or far_off):
            return 0.0
        
        reward = distance_reward
        return reward

    # Initializes the points used for determining the starting point of the vehicle
    def __init_road_points(self):
        self.__road_points = []
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'road_lines.txt'), 'r') as f:
            for line in f: # e.g: -255.21722656,-209.60329102   -127.91722656,-209.60329102
                # splits a string into a list with the separator: 
                points = line.split('\t') # e.g: ['-255.21722656,-209.60329102' , '-127.91722656,-209.60329102']
                first_point = np.array([float(p) for p in points[0].split(',')] + [0]) # e.g: [-255.21722656 -209.60329102  0.0]
                second_point = np.array([float(p) for p in points[1].split(',')] + [0]) # e.g: [-127.91722656 -209.60329102 0.0]
                self.__road_points.append(tuple((first_point, second_point))) # e.g: [(array([-255.21722656, -209.60329102, 0.]), array([-127.91722656 -209.60329102, 0.])), ... , ... ]

    # Initializes the points used for determining the optimal position of the vehicle during the reward function
    def __init_reward_points(self):
        self.__reward_points = []
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'reward_points.txt'), 'r') as f:
            for line in f: # e.g: -251.21722655999997	-209.60329102	-132.21722655999997	-209.60329102
                point_values = line.split('\t') # e.g: ['-251.21722655999997', '-209.60329102', '-132.21722655999997', '-209.60329102']
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0]) # e.g: [-251.21722656 -209.60329102    0.        ]
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0]) # e.g: [-132.21722656 -209.60329102    0.        ]
                self.__reward_points.append(tuple((first_point, second_point))) # e.g: [(array([-251.21722656, -209.60329102,    0.0]), array([-132.21722656, -209.60329102,    0.0])) , ... , ...]

    # Randomly selects a starting point on the road
    # Used for initializing an iteration of data generation from AirSim
    def __get_next_starting_point(self):
    
        # Get the current state of the vehicle
        car_state = self.__car_client.getCarState()

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(self.__road_points)) # len(self.__road_points)=15
        
        # Pick a random position on the road. 
        # Do not start too close to the end roads, as the car may crash immediately.
        # random floats in the half-open interval [0.3, 0.7)
        random_interp = (np.random.random_sample() * 0.4) + 0.3
        
        # Pick a random direction to face
        # random floats in the half-open interval [0.0, 1.0)
        random_direction_interp = np.random.random_sample()

        # Compute the starting point of the car
        random_line = self.__road_points[random_line_index] # e.g: (-255.21722656  -209.60329102  0.0    ,   -127.91722656  -209.60329102  0.0 )
        random_start_point = list(random_line[0]) # e.g: [-255.21722656, -209.60329102, 0.0]
        random_start_point[0] += (random_line[1][0] - random_line[0][0])*random_interp # e.g: -255.21722656 += (-127.91722656 - (-255.21722656))*random[0.3, 0.7)
        random_start_point[1] += (random_line[1][1] - random_line[0][1])*random_interp # e.g: -209.60329102 += (-209.60329102 - (-209.60329102))*random[0.3, 0.7)

        # Compute the direction that the vehicle will face
        # Horizontal line
        # np.isclose :Returns a boolean array where two arrays are element-wise equal within a relative & absolute tolerance.
        if (np.isclose(random_line[0][1], random_line[1][1])): # e.g:same Y = -209.60329102 so face x direction 0 or pi
            if (random_direction_interp > 0.5):
                random_direction = (0,0,0)
            else:
                random_direction = (0, 0, math.pi)
        # Vertical line
        elif (np.isclose(random_line[0][0], random_line[1][0])): # e.g:same X = -255.21722656 so face y direction pi/2 or -pi/2
            if (random_direction_interp > 0.5):
                random_direction = (0,0,math.pi/2)
            else:
                random_direction = (0,0,-1.0 * math.pi/2)

        # The z coordinate is always zero
        random_start_point[2] = -0
        return (random_start_point, random_direction)


# Parse the command line parameters
# weights_path = os.path.join(os.getcwd(), 'D:/Documents/spyder/DQNProject/DataDir/data/؟؟؟؟؟؟؟؟؟؟')
weights_path = None
train_from_json = 'false'
train_conv_layers = 'false'
data_dir = os.path.join(os.getcwd(), 'DataDir')

# How often to update the target network
batch_update_frequency = 300
#batch_update_frequency = 10

max_episode_runtime_sec = 30
per_iter_epsilon_reduction=0.003
min_epsilon = 0.05
batch_size = 32

# Note: The Deepmind paper suggests 1000000 however this causes memory issues
replay_memory_size = 3000
# replay_memory_size = 200

#Define some constant parameters for the reward function
THRESH_DIST_rst = 3.5                # The maximum distance from the center of the road to reset
THRESH_DIST_far = 1.5                # The maximum distance from the center of the road to compute the reward function
DISTANCE_DECAY_RATE = 1.3        # The rate at which the reward decays for the distance function

parameters = {}
parameters['batch_update_frequency'] = batch_update_frequency
parameters['max_episode_runtime_sec'] = max_episode_runtime_sec
parameters['per_iter_epsilon_reduction'] = per_iter_epsilon_reduction
parameters['min_epsilon'] = min_epsilon
parameters['batch_size'] = batch_size
parameters['replay_memory_size'] = replay_memory_size
parameters['weights_path'] = weights_path
parameters['train_from_json'] = train_from_json
parameters['train_conv_layers'] = train_conv_layers
parameters['data_dir'] = data_dir

#Make the debug statements easier to read
np.set_printoptions(threshold=sys.maxsize, suppress=True)

# Check additional parameters needed for run
if 'batch_update_frequency' not in parameters:
    print('ERROR: batch_update_frequency must be defined.')
    print('Please provide the path to airsim in a parameter like "batch_update_frequency=<int>"')
    sys.exit()


print('------------STARTING AGENT----------------')
print(parameters)

print('***')
print(os.environ)
print('***')

d = {'episode number' : [0], 'reward sum' : [0], 'reward avg' : [0], 'number of actions' : [0], 'episode duration' : [0], 'percent random actiions' : [0]}
df = pd.DataFrame( data = d )
df.set_index( 'episode number', inplace = True )
# export DataFrame to csv
df.to_csv( 'log.csv' )

# Start the training
agent = Agent(parameters)
agent.start()
