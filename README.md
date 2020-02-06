# Deep-Q-Network-TensorFlow1
Implementation of the DQN DeepMind research paper on "Playing Atari with Deep Reinforcement Learning". 
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

BreakoutNoFrameskip-v4 environment

Model:
- Convolutional layer 1: 32 filters, 8 × 8, stride 4, padding “SAME”, rectified linear activations.
- Convolutional layer 2: 64 filters, 4 × 4, stride 2, padding “SAME”, rectified linear activations.
- Convolutional layer 3: 64 filters, 3 × 3, stride 1, padding “SAME”, rectified linear activations.
- Fully connected layer 1: 512 rectified linear units.
- Fully connected layer 2: k linear units, where k is the number of actions.