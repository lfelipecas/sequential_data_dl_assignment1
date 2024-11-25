# Human Activity and Audio Recognition Project

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
3. [Problem Description](#problem-description)
4. [Proposed Solution](#proposed-solution)
   - [Human Activity Recognition (HAR)](#human-activity-recognition-har)
   - [Audio Recognition](#audio-recognition)
5. [Results](#results)
6. [Conclusions](#conclusions)
7. [Repository Structure](#repository-structure)
8. [How to Run](#how-to-run)

## Introduction

In recent years, the rapid advancements in deep learning and artificial intelligence have enabled innovative solutions to traditionally complex problems. This repository showcases an exploration of two pivotal tasks: **Audio Recognition** and **Human Activity Recognition (HAR)**, both of which leverage neural network architectures and data preprocessing techniques.

### Audio Recognition
The goal of the audio recognition system is to classify different **mathematical operations**—such as addition, subtraction, and multiplication—based on short audio recordings. This problem finds real-world applications in voice-based educational tools, assistive technologies, and intelligent tutoring systems. By transforming raw audio signals into spectrograms, the system effectively captures time-frequency characteristics of the input, enabling robust classification using deep neural networks.

### Human Activity Recognition (HAR)
The HAR task focuses on recognizing various human physical activities—such as jogging, walking, and jumping—using time-series data from accelerometers. Applications of HAR range from fitness tracking and healthcare monitoring to gesture recognition in human-computer interaction systems. Our solution processes accelerometer signals into structured input formats and employs advanced neural architectures to achieve highly accurate predictions.

This repository combines theoretical depth and practical implementation to solve these challenges. The project is designed to be a complete workflow, from raw data collection to model training and evaluation, offering a hands-on demonstration of deep learning's power and versatility. Furthermore, we provide detailed explanations of the data preprocessing steps, neural network architectures, and experimental results to facilitate learning and reproducibility.

By the end of this document, you will find:  
- Detailed explanations of the preprocessing and modeling techniques applied to audio recognition and HAR tasks.  
- Clear descriptions of the neural network architectures used (MLP, CNN, RNN, and LSTM) and the reasoning behind their selection.  
- Results that serve as a baseline for further experimentation and improvement.  

## Theoretical Framework

### Deep Learning in Human Activity Recognition (HAR)

Human Activity Recognition (HAR) involves the automatic identification of human movements using data from wearable sensors. The primary challenge in HAR is the temporal nature of the data, which requires the model to recognize patterns that evolve over time, often in the presence of sensor noise. Deep learning techniques provide powerful tools for addressing these challenges:  
- **Convolutional Neural Networks (CNNs)**: While traditionally used for image analysis, CNNs can identify local spatial patterns in accelerometer signals. For HAR, CNNs are effective in recognizing movement features by analyzing data as windows of activity.  
- **Recurrent Neural Networks (RNNs)**: Designed specifically for sequential data, RNNs analyze time steps one at a time, maintaining an internal state to capture temporal dependencies. They are well-suited for identifying the order and duration of movements but are prone to issues such as vanishing gradients.  
- **Long Short-Term Memory (LSTM)**: A specialized form of RNN, LSTM networks introduce gating mechanisms to learn long-term dependencies without being hindered by vanishing gradients. This makes them highly effective for HAR tasks where movements unfold over extended time periods.

### Deep Learning in Audio Recognition

Audio recognition leverages the properties of sound signals, such as frequency and amplitude, to classify or understand audio content. In this project, the audio data is transformed into **spectrograms**, which convert sound into 2D visual representations that encode time on one axis and frequency on the other. This enables the use of image-based deep learning models:  
- **Convolutional Neural Networks (CNNs)**: By treating spectrograms as images, CNNs can extract spatial patterns, such as variations in pitch and intensity, which are critical for distinguishing different audio commands.  

### Neural Network Architectures

1. **Multilayer Perceptron (MLP)**:  
   - The MLP is the simplest form of neural network, composed of fully connected layers.  
   - It requires input data to be flattened into one-dimensional vectors, losing spatial or sequential structure.  
   - While effective for capturing generalized patterns, it lacks the ability to exploit relationships in time-series or image-like data.  

2. **Convolutional Neural Networks (CNNs)**:  
   - CNNs are specifically designed to extract local spatial features using convolutional filters.  
   - They are highly efficient for tasks involving structured input, such as spectrograms in audio recognition or windowed accelerometer data in HAR.  
   - Features are learned hierarchically, with deeper layers capturing increasingly abstract patterns.  

3. **Recurrent Neural Networks (RNNs)**:  
   - RNNs process sequential data by maintaining an internal "memory" of prior time steps.  
   - This memory enables the network to consider the order of inputs, making it particularly useful for tasks like HAR, where activities unfold over time.  
   - Standard RNNs, however, struggle with learning long-term dependencies due to vanishing gradients during training.  

4. **Long Short-Term Memory (LSTM)**:  
   - LSTMs are a sophisticated variant of RNNs designed to mitigate the vanishing gradient problem.  
   - They incorporate "gates" to regulate the flow of information, deciding what to keep, update, or forget.  
   - This makes LSTMs capable of learning long-term dependencies, essential for understanding extended sequences in both HAR and audio recognition tasks.

### Summary

This framework outlines the theoretical principles behind the methodologies applied in this project. The combination of temporal and spatial data analysis, alongside tailored neural architectures, allows for robust modeling of audio and human activity signals. The choice of architecture is driven by the unique requirements of each domain, ensuring both practical and theoretical rigor.

## Problem Description

### Human Activity Recognition (HAR)
The dataset comprises accelerometer signals collected from a smartphone placed alternately in the **left and right hands** while performing six distinct physical activities:
1. **Shake side-to-side** (*agitar a los lados*).  
2. **Shake up and down** (*agitar arriba abajo*).  
3. **Jump** (*brincar*).  
4. **Walk** (*caminar*).  
5. **Move in circles** (*círculos*).  
6. **Jog** (*trotar*).  

The goal is to classify these activities using deep learning models that can effectively learn from temporal and spatial patterns in the data. The challenge includes handling sensor noise and variability due to different hands being used during data collection.

### Audio Recognition
The dataset contains audio commands related to mathematical operations recorded in **Spanish**, such as:  
- *"dos más nueve"* ("two plus nine").  
- *"cuatro menos cinco"* ("four minus five").  
- *"tres por siete"* ("three times seven").  
- *"cinco dividido entre ocho"* ("five divided by eight").  
- *"raíz cuadrada de siete"* ("square root of seven").  
- *"cero elevado a la tres"* ("zero to the power of three").  

These commands are provided as `.wav` files. The task involves preprocessing the audio data, transforming it into meaningful representations (e.g., spectrograms), and using these representations to classify the spoken command. The inclusion of the Spanish language adds a layer of specificity, requiring the model to recognize phonetic nuances of mathematical terms in Spanish.

## Proposed Solution

### Human Activity Recognition (HAR)

#### Data Preprocessing
To prepare accelerometer signals for training, a series of preprocessing steps were implemented:

1. **Normalization**:  
   All accelerometer signals were normalized to a range of [0, 1] using `MinMaxScaler`. This ensures that the model does not favor signals with higher magnitudes and helps stabilize training.

2. **Windowing**:  
   The continuous time-series data was segmented into fixed-size windows of 104 samples (approximately 1 second of data). This windowing captures local temporal patterns while maintaining computational efficiency.

3. **Label Alignment**:  
   Each window was assigned a label based on the majority activity in the window, ensuring precise alignment between data segments and corresponding activities.

#### Neural Network Architectures and Training

**1. Multilayer Perceptron (MLP)**:  
   The MLP serves as a baseline model for HAR. Its architecture includes:  
   - **Input Layer**: Accepts flattened feature vectors from each time-series window.  
   - **Hidden Layers**: Three fully connected layers with 256, 128, and 64 neurons, respectively, each using ReLU activation.  
   - **Output Layer**: A dense layer with six neurons (corresponding to the six activities) and a softmax activation function for multi-class classification.

   **Training Configuration**:  
   - **Optimizer**: Adam  
   - **Loss Function**: Categorical Cross-Entropy  
   - **Epochs**: 60  
   - **Batch Size**: 32  

   The MLP, while effective at learning basic patterns, lacks spatial or temporal pattern recognition, limiting its performance compared to more advanced models.

**2. Convolutional Neural Network (CNN)**:  
   The CNN extracts spatial features directly from the accelerometer signals:  
   - **Input Layer**: Accepts 3D tensors (window_size x features).  
   - **Convolutional Layers**: Three convolutional layers with 16, 32, and 64 filters, each with a kernel size of 3. These layers capture local spatial patterns across time.  
   - **Pooling Layer**: A MaxPooling layer reduces dimensionality while preserving critical features.  
   - **Flatten Layer**: Converts the feature maps into a single vector for the dense layers.  
   - **Dense Layers**: Includes one fully connected layer (128 neurons, ReLU activation) followed by an output layer with six neurons (softmax activation).

   **Training Configuration**:  
   - Same as the MLP.  

   The CNN's spatial feature extraction enhances its ability to recognize activities from sensor data, outperforming the MLP.

**3. Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)**:  
   Both RNN and LSTM are tailored for sequential data.  
   - **Input Layer**: Processes sequential data directly.  
   - **RNN**: Two SimpleRNN layers (128 and 64 neurons, ReLU activation) to capture temporal dependencies.  
   - **LSTM**: Two LSTM layers (128 and 64 neurons, ReLU activation) address the vanishing gradient problem and preserve long-term dependencies.  
   - **Output Layer**: Similar to the CNN and MLP.

   **Training Configuration**:  
   - Identical to MLP and CNN.

   While RNNs capture short-term dependencies effectively, LSTMs offer superior performance by addressing longer-term temporal relationships. However, they are computationally more intensive.

### Audio Recognition

#### Data Preprocessing

1. **Waveform Preparation**:  
   - `.wav` files were loaded and resampled to a mono channel at 16kHz.  
   - Waveforms were truncated or zero-padded to ensure a consistent duration of 3 seconds.

2. **Spectrogram Generation**:  
   - Short-Time Fourier Transform (STFT) was applied to each waveform to generate spectrograms, converting the 1D audio signals into 2D time-frequency representations.  
   - The spectrogram values represented the magnitude of energy at each frequency over time.

#### Neural Network Architecture and Training

**Model 1: Baseline CNN**  
   - **Input Layer**: Accepts spectrograms in their natural shape derived from STFT output.  
   - **Convolutional Layers**: Two convolutional layers with 32 and 64 filters respectively, each with a kernel size of `(3, 3)` and ReLU activation.  
   - **Pooling Layers**: MaxPooling layers `(2, 2)` follow each convolution, reducing dimensionality while preserving critical spatial patterns.  
   - **Flatten Layer**: Converts 2D feature maps into a single vector for classification.  
   - **Dense Layers**:  
      - A dense layer with 128 neurons and ReLU activation.  
      - An output layer with six neurons and softmax activation for classification.

**Model 2: Enhanced CNN**  
   - **Input Layer**: Same as Model 1.  
   - **Convolutional Layers**:  
      - Two convolutional layers with 32 and 64 filters, each followed by `BatchNormalization` to stabilize training and improve performance.  
   - **Pooling Layers**: Similar to Model 1, with added `Dropout` (0.3) layers for regularization.  
   - **Dense Layers**:  
      - A dense layer with 256 neurons (larger capacity for richer feature representation) and ReLU activation, followed by a `Dropout` (0.4).  
      - An output layer with six neurons and softmax activation.  

**Training Configuration**:  
   - **Epochs**: 20 for both models.  
   - **Batch Size**: 16.  
   - **Optimizer**: Adam.  
   - **Loss Function**: Categorical Cross-Entropy.  

## Results

### Human Activity Recognition
**Model Performance**:
- **MLP**: 84.72% Accuracy, Loss = 0.3054
- **CNN**: 97.22% Accuracy, Loss = 0.0731
- **RNN**: 40.28% Accuracy, Loss = 1.0988
- **LSTM**: 33.33% Accuracy, Loss = 1.4949

**Confusion Matrices**:
![MLP Confusion Matrix](output_graphs/mlp_training_loss.png)
![CNN Confusion Matrix](output_graphs/cnn_training_loss.png)
![RNN Confusion Matrix](output_graphs/rnn_training_loss.png)
![LSTM Confusion Matrix](output_graphs/lstm_training_loss.png)

### Audio Recognition
- **CNN Accuracy**: 97.1%, Loss = 0.0673

**Spectrogram Training Loss**:
![Spectrogram Training Loss](output_graphs/sample_spectrogram.png)

## Conclusions

1. **Human Activity Recognition**:
   - CNNs performed best due to their ability to detect local patterns in accelerometer signals.
   - RNN and LSTM models underperformed, possibly due to insufficient training or hyperparameter optimization for sequential data.
   - The high accuracy of CNNs (97.22%) suggests they are well-suited for HAR tasks.

2. **Audio Recognition**:
   - Treating audio as images (spectrograms) allowed CNNs to achieve near-perfect accuracy (97.1%).
   - This approach is robust, as it captures both time and frequency-domain information effectively.

3. **Future Work**:
   - Experiment with GRU or transformer-based architectures for HAR to improve temporal pattern recognition.
   - Investigate ensemble methods combining CNNs and RNNs for audio tasks.

## Repository Structure
```
│   requirements.txt
│   .gitignore
│   README.md
│   
├───data
│   ├───mathematical_operations
│   │   │   info.labels
│   │   │   README.txt
│   │   │   
│   │   ├───testing
│   │   │       division1.wav
│   │   │       division2.wav
│   │   │       division3.wav
│   │   │       info1.labels
│   │   │       multiplicacion1.wav
│   │   │       multiplicacion2.wav
│   │   │       multiplicacion3.wav
│   │   │       potenciacion1.wav
│   │   │       potenciacion2.wav
│   │   │       potenciacion3.wav
│   │   │       raizcuadrada1.wav
│   │   │       raizcuadrada2.wav
│   │   │       raizcuadrada3.wav
│   │   │       resta1.wav
│   │   │       resta2.wav
│   │   │       resta3.wav
│   │   │       suma1.wav
│   │   │       suma2.wav
│   │   │       suma3.wav
│   │   │
│   │   └───training
│   │           division1.wav
│   │           division2.wav
│   │           division3.wav
│   │           division4.wav
│   │           division5.wav
│   │           division6.wav
│   │           division7.wav
│   │           division8.wav
│   │           division9.wav
│   │           division10.wav
│   │           division11.wav
│   │           division12.wav
│   │           division13.wav
│   │           info1.labels
│   │           multiplicacion1.wav
│   │           multiplicacion2.wav
│   │           multiplicacion3.wav
│   │           multiplicacion4.wav
│   │           multiplicacion5.wav
│   │           multiplicacion6.wav
│   │           multiplicacion7.wav
│   │           multiplicacion8.wav
│   │           multiplicacion9.wav
│   │           multiplicacion10.wav
│   │           multiplicacion11.wav
│   │           multiplicacion12.wav
│   │           multiplicacion13.wav
│   │           potenciacion1.wav
│   │           potenciacion2.wav
│   │           potenciacion3.wav
│   │           potenciacion4.wav
│   │           potenciacion5.wav
│   │           potenciacion6.wav
│   │           potenciacion7.wav
│   │           potenciacion8.wav
│   │           potenciacion9.wav
│   │           potenciacion10.wav
│   │           potenciacion11.wav
│   │           potenciacion12.wav
│   │           potenciacion13.wav
│   │           raizcuadrada1.wav
│   │           raizcuadrada2.wav
│   │           raizcuadrada3.wav
│   │           raizcuadrada4.wav
│   │           raizcuadrada5.wav
│   │           raizcuadrada6.wav
│   │           raizcuadrada7.wav
│   │           raizcuadrada8.wav
│   │           raizcuadrada9.wav
│   │           raizcuadrada10.wav
│   │           raizcuadrada11.wav
│   │           raizcuadrada12.wav
│   │           raizcuadrada13.wav
│   │           resta1.wav
│   │           resta2.wav
│   │           resta3.wav
│   │           resta4.wav
│   │           resta5.wav
│   │           resta6.wav
│   │           resta7.wav
│   │           resta8.wav
│   │           resta9.wav
│   │           resta10.wav
│   │           resta11.wav
│   │           resta12.wav
│   │           resta13.wav
│   │           suma1.wav
│   │           suma2.wav
│   │           suma3.wav
│   │           suma4.wav
│   │           suma5.wav
│   │           suma6.wav
│   │           suma7.wav
│   │           suma8.wav
│   │           suma9.wav
│   │           suma10.wav
│   │           suma11.wav
│   │           suma12.wav
│   │           suma13.wav
│   │
│   └───movements
│       │   info.labels
│       │   README.txt
│       │
│       ├───testing
│       │       agitaraloslados1.json
│       │       agitaraloslados2.json
│       │       agitararribaabajo1.json
│       │       agitararribaabajo2.json
│       │       brincar1.json
│       │       brincar2.json
│       │       caminar1.json
│       │       caminar2.json
│       │       circulos1.json
│       │       circulos2.json
│       │       info.labels
│       │       trotar1.json
│       │       trotar2.json
│       │
│       └───training
│               agitaraloslados1.json
│               agitaraloslados2.json
│               agitaraloslados3.json
│               agitaraloslados4.json
│               agitaraloslados5.json
│               agitaraloslados6.json
│               agitaraloslados7.json
│               agitaraloslados8.json
│               agitararribaabajo1.json
│               agitararribaabajo2.json
│               agitararribaabajo3.json
│               agitararribaabajo4.json
│               agitararribaabajo5.json
│               agitararribaabajo6.json
│               agitararribaabajo7.json
│               agitararribaabajo8.json
│               brincar1.json
│               brincar2.json
│               brincar3.json
│               brincar4.json
│               brincar5.json
│               brincar6.json
│               brincar7.json
│               brincar8.json
│               caminar1.json
│               caminar2.json
│               caminar3.json
│               caminar4.json
│               caminar5.json
│               caminar6.json
│               caminar7.json
│               caminar8.json
│               circulos1.json
│               circulos2.json
│               circulos3.json
│               circulos4.json
│               circulos5.json
│               circulos6.json
│               circulos7.json
│               circulos8.json
│               info.labels
│               trotar1.json
│               trotar2.json
│               trotar3.json
│               trotar4.json
│               trotar5.json
│               trotar6.json
│               trotar7.json
│               trotar8.json
│
├───notebooks
│   │   audio_recognition.ipynb
│   │   human_activity_recognition.ipynb
│   │   rename_files.ipynb
│   │   audio_recognition.py
│   │   human_activity_recognition.py
│   │
│   ├───__pycache__
│   │       utils.cpython-311.pyc
│   │
│   └───output_graphs
│           sample_spectrogram.png
│
├───output_audios
│       division.wav
│       multiplicacion.wav
│       potenciacion.wav
│       raizcuadrada.wav
│       resta.wav
│       suma.wav
│
└───output_graphs
        agitaraloslados_accelerometer_signals.png
        agitararribaabajo_accelerometer_signals.png
        brincar_accelerometer_signals.png
        caminar_accelerometer_signals.png
        circulos_accelerometer_signals.png
        trotar_accelerometer_signals.png
        combined_accelerometer_signals.png
        normalized_combined_accelerometer_signals.png
        normalized_random_window_signals.png
        mlp_training_loss.png
        cnn_training_loss.png
        rnn_training_loss.png
        lstm_training_loss.png
        sample_waveform.png
        sample_spectrogram.png
```

## How to Run

1. **Clone Repository**:
   ```
   git clone https://github.com/lfelipecas/sequential_data_dl_assignment1
   ```
2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Run Notebooks**:
   - `notebooks/audio_recognition.ipynb`: Audio recognition pipeline.
   - `notebooks/human_activity_recognition.ipynb`: HAR pipeline.
4. **View Results**:
   - Plots and confusion matrices are saved in the `output_graphs/` directory.