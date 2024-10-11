# My Projects Portfolio

Welcome to my portfolio! Here, I showcase my key projects, detailing the skills I applied, the methods I used, and the technical details behind each project. Explore my work to see my expertise in action!

---

## 1. **Vertical Lift Bridge Simulation**

**Project Description**:  
This project simulates the behavior of a vertical lift bridge, including real-time dynamic responses based on external inputs like traffic load and environmental factors.

### Skills:
- Python
- Simulation and Modeling
- Computer Vision

### How I Did It:
- Modeled the bridge dynamics using Python’s physics libraries.
- Implemented real-time simulation, considering external conditions like wind and weight load.
- Integrated a GUI for user interaction.

### Project Code Description:
- **Simulation Module**: Models the vertical movement and mechanical behavior of the bridge.
- **User Interface**: A simple interface to control the bridge operation and visualize real-time simulations.

### Dataset:
- **Size**: Synthetic dataset of 10,000 bridge operation scenarios.
- **Preprocessing**: Cleaned and standardized input data to simulate traffic load and weather conditions.

### Model Building:
- Developed a simulation engine using finite element analysis (FEA) for bridge movements.
- Implemented control systems to optimize lift operations based on traffic and weather data.

### Testing:
- Simulated various real-world conditions and measured bridge response times.
- Validated with engineering datasets of real bridge movements.

---

## 2. **AI Board Game**

**Project Description**:  
An AI-driven board game that autonomously solves game dynamics, predicting player moves and optimizing strategies using reinforcement learning.

### Skills:
- Reinforcement Learning
- Python (PyTorch, TensorFlow)
- Game AI

### How I Did It:
- Created a game environment where the AI learns from playing multiple games.
- Designed reward systems for optimizing decision-making in a turn-based strategy game.
- Applied reinforcement learning techniques to improve the AI over time.

### Project Code Description:
- **RL Agent**: Implemented using the Proximal Policy Optimization (PPO) algorithm.
- **Game Environment**: Custom game environment compatible with OpenAI’s Gym.

### Dataset:
- **Size**: Simulated 100,000 game rounds to train the agent.
- **Preprocessing**: Extracted key game states and rewards for training reinforcement learning models.

### Model Building:
- Built a deep neural network with policy gradients to predict optimal moves.
- Integrated a value network to estimate future rewards from game states.

### Testing:
- Tested AI against human players and other AI agents.
- Evaluated performance using win/loss ratios and learning curves.

---

## 3. **Term Deposit Decision-Making Using ANN**

**Project Description**:  
A machine learning model to predict whether a customer will subscribe to a term deposit based on banking data. The project uses an Artificial Neural Network (ANN) for classification.

### Skills:
- Artificial Neural Networks (ANN)
- Python (Scikit-Learn, Keras)
- Data Analysis

### How I Did It:
- Collected a large dataset of customer banking information.
- Applied data preprocessing steps such as feature scaling and one-hot encoding for categorical data.
- Trained an ANN model for binary classification.

### Project Code Description:
- **Data Preprocessing Module**: Handles feature scaling, missing values, and one-hot encoding.
- **Model Module**: Defines and trains an ANN for predicting term deposit subscription.

### Dataset:
- **Size**: 40,000 customer records from a bank marketing dataset.
- **Preprocessing**: Cleaned the data by removing missing values and scaling numeric features.

### Model Building:
- Constructed an ANN with 3 hidden layers using the Keras library.
- Optimized using the Adam optimizer and binary cross-entropy loss function.

### Testing:
- Split data into training and testing sets (80/20 split).
- Achieved 85% accuracy on the test set, with precision and recall above 80%.

---

## 4. **Nail Disease Prediction Using Vision Transformer (ViT)**

**Project Description**:  
This project focuses on building a predictive model for nail diseases using images, leveraging MobileNet for feature extraction and a pre-trained Vision Transformer (ViT) for classification.

### Skills:
- Deep Learning (MobileNet, Vision Transformer)
- Image Processing
- PyTorch

### How I Did It:
- Preprocessed nail disease image datasets by resizing and normalizing the images.
- Extracted features using MobileNet and used those features for classification using ViT.

### Project Code Description:
- **Feature Extraction Module**: Uses MobileNet for extracting image features.
- **Classification Module**: A pre-trained ViT fine-tuned for nail disease classification.

### Dataset:
- **Size**: 10,000 images of different nail diseases.
- **Preprocessing**: Resized all images to 224x224, normalized pixel values, and augmented the dataset with flips and rotations.

### Model Building:
- Fine-tuned a pre-trained ViT model for image classification.
- Combined MobileNet as a feature extractor to enhance performance.

### Testing:
- Used 5-fold cross-validation to evaluate model robustness.
- Achieved an F1-score of 0.92, with high precision for disease categories.

---

## 5. **Admittance Matrix Identification Using Meta-Heuristics**

**Project Description**:  
Optimization problem involving the least squares identification of the admittance matrix of a power system using meta-heuristic techniques.

### Skills:
- Meta-Heuristics (PSO, GA)
- Least Squares Optimization
- Power Systems

### How I Did It:
- Formulated the admittance matrix identification as a least squares optimization problem.
- Applied Particle Swarm Optimization (PSO) and Genetic Algorithms (GA) to find the optimal matrix.

### Project Code Description:
- **Optimization Module**: Implements PSO and GA for matrix optimization.
- **Error Calculation**: Computes the least squares error to evaluate the fitness of each solution.

### Dataset:
- **Size**: Simulated data from a 20-bus power system.
- **Preprocessing**: Normalized power measurements and calculated target values.

### Model Building:
- Developed fitness functions to evaluate solutions based on the least squares error.
- Implemented constraints to ensure matrix validity for power system stability.

### Testing:
- Tested the optimization algorithms on multiple test cases.
- Achieved accurate identification with less than 2% error in the matrix.

---

### Contact Information:
- **Email**: yourname@example.com
- **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
