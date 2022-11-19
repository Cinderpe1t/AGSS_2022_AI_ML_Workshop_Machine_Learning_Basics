# AGSS 2022 AI ML - Machine Learning Basics
AGSS 2022 AI ML - Machine Learning Basics
Let's learn neural network machine learning basics
## Supervised neural network machine learning by error back propagation
- `agss_4_1_ml_introduction_propagation_v1.py`
- ![propagation screen](https://github.com/Cinderpe1t/AGSS_2022_AI_ML_Workshop_Machine_Learning_Basics/blob/main/agss_4_1_ml_propagation.png)
- Neural network learns by comparing input and desired output, and correcting its connections
- Feed forward propagation for evaluation: input is processed by the weight of the neural network connection and summed
- Feed backward error propagation: Any error from the known answer is fed backward to update neural network connection weights
- It is called re-enforced learning, as the answer vs. input is enforced with forward and backward propagation
- Think about teaching your dog or dragon with snack
## Neural network machine learning by repetition or epochs
- `agss_4_2_ml_introduction_epoch_v1.py`
- ![epochs](https://github.com/Cinderpe1t/AGSS_2022_AI_ML_Workshop_Machine_Learning_Basics/blob/main/agss_4_2_ml_epoch.png)
- Does your dog learn tricks with first try?
- Machine needs lots of repetition to learn
- How much strong would like the machine learn? Change the learning rate.
- How many times would like the machine to repeat? Change the number of epochs.
## Neural network machine learning with fashion items classification
- `agss_4_3_ml_classification.py`
- ![classification](https://github.com/Cinderpe1t/AGSS_2022_AI_ML_Workshop_Machine_Learning_Basics/blob/main/agss_4_3_ml_classification.png)
- The same TensorFlow example is found at [TensorFlow basic classification example](https://www.tensorflow.org/tutorials/keras/classification) 
- This example was implemented with NumPy to avoid TensorFlow installation problems.
- Data file 'train_images.npy' size is too big even after compression to upload to Github (25MB limit)
- You can make train and test data from Tensorflow example
```
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
- If you can configure Tensorflow, running through the classification tutorial will bring you to the equivalent demonstration.
- There are 60,000 training clothing items with ten item labels.
- Each image has 28x28=768 pixels
- There are 10,000 evaluation clothing images to evaluate neural network training accuracy
- You can change neural network number of layers and size at each layer
- Competition: which team would make best trained machine in given time?
