# Towards-Self-driving-Cars-using-Prioritized-Double-DQN-in-Python
Deep Reinforcement Learning
## Prerequisites and setup
### Background needed
* basic concepts of reinforcement learning. A helpful introduction to reinforcement learning can be [found here by Professor Emma Brunskill, Stanford University](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u) you'll get DQN in lecture 6
* I also highly recommend [this course by Professor Yann LeCun, winner of 2018 ACM A.M. Turing Award, and Alfredo Canziani, NYU](https://www.youtube.com/playlist?list=PLLHTzKZzVU9e6xUfG10TkTWApKSZCzuBI) on Deep Learning
* you can find more on the Internet but remember "Stand on the Shoulders of Giants"
## My Envinronment Setup
* Anaconda
* Keras v2.4.3 on Tensorflow v2.3 & CUDA 10
## Simulator
[Airsim](https://microsoft.github.io/AirSim/) in the landscape environment
### Notice
I used an older version of Airsim API, so I don't guarantee this works fine with the latest version.
## Note
In this project, I suggest you use the transfer learning method to first train the Conv Layers and then just train the subsequent Fully Connected Layers using Reinforcement Learning methods like Prioritized Double DQN to reduce the trainig time. I personally prefer this way, but your agent could also learn both vision and driving abilities from scratch, which considerably takes longer (depending on the number and Computing Capability of your GPAs).
## What next
Computer Vision for Autonomous Vehicles
