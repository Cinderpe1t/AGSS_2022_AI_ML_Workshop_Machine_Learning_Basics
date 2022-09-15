import time
import tkinter as tk
import random
import numpy as np
import math
import copy

#Load Fashion MNIST data saved offline
train_images=np.load('train_images.npy')
test_images=np.load('test_images.npy')
train_labels=np.load('train_labels.npy')
test_labels=np.load('test_labels.npy')

train_images = train_images / 255.0
test_images = test_images / 255.0

train_matrix = train_images.reshape(60000,train_images.shape[1]*train_images.shape[2])
test_matrix = test_images.reshape(10000,train_images.shape[1]*train_images.shape[2])

train_output = np.zeros((60000,10))
for k in range(60000):
    tI=train_labels[k]
    train_output[k][tI]=1

test_output = np.zeros((10000,10))
for k in range(10000):
    tI=test_labels[k]
    test_output[k][tI]=1

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

tCX=np.transpose(train_matrix)
tCY=np.transpose(train_output)
tCXTest=np.transpose(test_matrix)
tCYTest=np.transpose(test_output)

def cAnimFaster():
    global stepDelay
    if stepDelay>0:
        stepDelay=round(100*(stepDelay-0.1))/100
        inputSpeed.delete('1.0', tk.END)
        inputSpeed.insert(tk.END, stepDelay)
        
def cAnimSlower():
    global stepDelay
    stepDelay=round(100*(stepDelay+0.1))/100
    inputSpeed.delete('1.0', tk.END)
    inputSpeed.insert(tk.END, stepDelay)

def cAnimationDelay():
    global stepDelay
    inp = inputSpeed.get(1.0, "end-1c")
    stepDelay = float(inp)

def cTrain():
    global tNoEpoch, tLearnRate
    inp = inputNoEpoch.get(1.0, "end-1c")
    tNoEpoch = int(inp)
    inp = inputLearnRate.get(1.0, "end-1c")
    tLearnRate = float(inp)

    if len(tCfgCNN)==0:
        cCNNInit()
        
    train()
    
def cEvaluate():
    global tCXTest, tCNN, tCfgCNN
    Y_hat, cashe = full_forward_propagation(tCXTest, tCNN, tCfgCNN)
    accuracy = get_accuracy_value(Y_hat, tCYTest)
    textEvaluateAccuracy.config(text = "Accuracy: {:.5f}".format(accuracy))
    win.update()
    
def cEvaluateRandom():
    global tCXTest, tCNN, tCfgCNN, tRObj
    #delete previous objects
    if len(tRObj)>0:
        for i in range(len(tRObj)):
            canvR.delete(tRObj[i])
            
    testSize=tCXTest.shape
    tI=np.random.randint(0,testSize[1])
    Y_hat, cashe = full_forward_propagation(np.transpose(test_matrix[[tI],:]), tCNN, tCfgCNN)
    tD=test_matrix[[tI],:]
    print(tD.shape)
    #print(Y_hat.shape)
    #accuracy = get_accuracy_value(Y_hat, np.transpose(test_output[[tI],:]))
    iGuess=np.argmax(Y_hat)
    iAnswer=np.argmax(test_output[[tI],:])
    maxProb=np.amax(Y_hat)
    
    textEvaluateRandom.config(text = "Guessed "+class_names[iGuess]+" for "+class_names[iAnswer])
    textEvaluateRandomProb.config(text = "Probability:  %.3f" % (maxProb))
    randomImage=test_images[tI]
    #print(randomImage.shape)
    xOffset=70
    yOffset=20
    pixelSize=6
    for i in range(randomImage.shape[0]):
        tX1=xOffset+pixelSize*i
        tX2=xOffset+pixelSize*(i+1)
        for j in range(randomImage.shape[1]):
            tY1=yOffset+pixelSize*j
            tY2=yOffset+pixelSize*(j+1)
            colorValue=255-int(255*randomImage[j][i])
            strColor=rgbToHex(colorValue,colorValue,colorValue)
            tO=canvR.create_rectangle(tX1,tY1,tX2,tY2,fill=strColor,outline='')
            tRObj.append(tO)    
    
    win.update()

def cCheck():
    global tObj, mouseXY, tCoord, tGameStat, tLastMove, tBoard, tNoGames, tP1LastMove, tP2LastMove
    tGameStat=[0,0,0,0]
    tPlayTurn=0
    
def cStop():
    global tStop
    tStop=1

def cCNNInit():
    global tCfgCNNSize, tCfgCNN, tCNN, tGuess, tCNNAccuracy, tAccumulatedEpoch
    inputString = inputNetworkArch.get(1.0, "end-1c")
    tCfgCNNSize=[28*28] #size of image
    tCNNAccuracy=[]
    tGuess=[]
    tAccumulatedEpoch=0
    
    ListString=inputString.split('\n')
    for i in range(len(ListString)):
        if len(ListString[i])>0:
            tCfgCNNSize.append(int(ListString[i]))
    tCfgCNNSize.append(10) #last output

    tCfgCNN=[]
    for i in range(len(tCfgCNNSize)-1):
        if i<len(tCfgCNNSize)-2:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "relu"})
        else:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "sigmoid"})

    tCNN=init_layers(tCfgCNN, seed = 99)
    
    drawCNN()
    displayNetworkImage()
    drawAccuracy()
    win.update()


def rgbToHex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

def drawCNN():
    global tCfgCNNSize, tCfgCNN, tCNN, tCObj
    #clear if there is existing drawing
    if len(tCObj)>0:
        for i in range(len(tCObj)):
            canvC.delete(tCObj[i])

    tYos=0.08*tCHeight
    tDY=0.84*tCHeight

    #Connections
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        #print(tW)
        tY1=k*tDY/(len(tCfgCNNSize)-1)+tYos
        tY2=(k+1)*tDY/(len(tCfgCNNSize)-1)+tYos
        for i in range(tCfgCNNSize[k]):
            tX1=(i+0.5)*tCWidth/(tCfgCNNSize[k])
            for j in range(tCfgCNNSize[k+1]):
                tX2=(j+0.5)*tCWidth/(tCfgCNNSize[k+1])
                if tW[j][i]>0:
                    colorValue=int(255*tW[j][i]/maxValue)
                    strColor=rgbToHex(255,255-colorValue,255-colorValue)
                elif tW[j][i]<0:
                    colorValue=abs(int(255*tW[j][i]/minValue))
                    strColor=rgbToHex(255-colorValue,255-colorValue,255)
                else:
                    strColor=rgbToHex(255,255,255)
                tO=canvC.create_line(tX1,tY1,tX2,tY2, width = 1, fill=strColor)
                tCObj.append(tO)

    #Draw nodes
    for i in range(len(tCfgCNNSize)):
        tY=i*tDY/(len(tCfgCNNSize)-1)+tYos
        for j in range(tCfgCNNSize[i]):
            tX=(j+0.5)*tCWidth/(tCfgCNNSize[i])
            tO=canvC.create_oval(tX-1,tY-1,tX+1,tY+1, width = tLineWidth, outline='black')
            tCObj.append(tO)


def displayNetworkImage():
    global tCfgCNNSize, tCfgCNN, tCNN, tIObj
    #delete previous objects
    if len(tIObj)>0:
        for i in range(len(tIObj)):
            canvI.delete(tIObj[i])

    tYos=0.03*tIHeight
    tDY=0.96*tIHeight
    
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        tY1=k*tDY/(len(tCfgCNNSize)-1)+tYos
        tY2=(k+1)*tDY/(len(tCfgCNNSize)-1)+tYos
        dY=(tY2-tY1)*0.9
        for i in range(tCfgCNNSize[k]):
            tX1=0.02*tIWidth
            tX2=0.98*tIWidth
            dX=tX2-tX1
            for j in range(tCfgCNNSize[k+1]):
                if tW[j][i]>0:
                    colorValue=int(255*tW[j][i]/maxValue)
                    strColor=rgbToHex(255,255-colorValue,255-colorValue)
                elif tW[j][i]<0:
                    colorValue=abs(int(255*tW[j][i]/minValue))
                    strColor=rgbToHex(255-colorValue,255-colorValue,255)
                else:
                    strColor=rgbToHex(255,255,255)
                ttX1=tX1+i*dX/tCfgCNNSize[k]
                ttX2=tX1+(i+1)*dX/tCfgCNNSize[k]
                ttY1=tY1+j*dY/tCfgCNNSize[k+1]
                ttY2=tY1+(j+1)*dY/tCfgCNNSize[k+1]
                tO=canvI.create_rectangle(ttX1,ttY1,ttX2,ttY2,fill=strColor,outline='')
                tIObj.append(tO)


def drawAccuracy():
    global tCNNAccuracy, tAObj, tAHeight, tOffset, tAWidth
    #clear if there is existing drawing
    #print("clean up")
    #win.update()
    #time.sleep(5)
    if len(tAObj)>0:
        for i in range(len(tAObj)):
            canvA.delete(tAObj[i])

    
    tOff=0.05*tAWidth

    #axis
    tO=canvA.create_line(tOff,tAHeight-tOff,tOff,tOff, width = 1, fill='black')
    tAObj.append(tO)
    tO=canvA.create_line(tOff,tAHeight-tOff,tAWidth-tOff,tAHeight-tOff, width = 1, fill='black')
    tAObj.append(tO)
    
    if len(tCNNAccuracy)==0:
        return
    if len(tCNNAccuracy)<50:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/100
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            
    else:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/len(tCNNAccuracy)
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            


#CNN functions
def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    
    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X
        
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def find_winner(probs):
    winner = np.zeros(probs.shape) 
    winner[np.argmax(probs, axis=0), np.arange(probs.shape[1])] = 1
    #print(probs)
    #print(winner)
    return winner

#def get_accuracy_value(Y_hat, Y):
#    Y_hat_ = convert_prob_into_class(Y_hat)
#    return (Y_hat_ == Y).all(axis=0).mean()

def get_accuracy_value(Y_hat, Y):
    global tGuess
    Y_hat_ = find_winner(Y_hat)
    #print(Y_hat_)
    #print(Y)
    #print(Y_hat_ == Y)
    #print((Y_hat_ == Y).all(axis=0))
    tGuess=(Y_hat_ == Y).all(axis=0)
    return (Y_hat_ == Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    
    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    
    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # derivative of the matrix A_prev
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[1]
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
    return params_values;

def train():
    global tNoEpoch, tLearnRate, tCNN, tCNNAccuracy, tStop, tAccumulatedEpoch
    #train(X, Y, nn_architecture, epochs, learning_rate, verbose=False, callback=None):
    #numpy cnn code from https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    # initiation of neural net parameters
    # params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    #cost_history = []
    #accuracy_history = []
    
    # performing calculations for subsequent iterations
    for i in range(tNoEpoch):
        # step forward
        Y_hat, cashe = full_forward_propagation(tCX, tCNN, tCfgCNN)
        
        # calculating metrics and saving them in history
        #cost = get_cost_value(Y_hat, Y)
        #cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, tCY)
        tCNNAccuracy.append(accuracy)
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, tCY, cashe, tCNN, tCfgCNN)
        # updating model state
        tCNN = update(tCNN, grads_values, tCfgCNN, tLearnRate)

        if(i % 20 == 0):
            drawCNN()
            displayNetworkImage()
            win.update()
            print("Iteration: {:05} - accuracy: {:.5f}".format(i, accuracy))

        if tStop==1:
            tStop=0
            textTrainingCount.config(text = "Epoch: %4d / %4d / %4d - stopped" % (i+1,tNoEpoch,tAccumulatedEpoch))
            return

        drawAccuracy()
        tAccumulatedEpoch+=1
        textTrainingCount.config(text = "Epoch: %4d / %4d / %4d" % (i+1,tNoEpoch,tAccumulatedEpoch))
        textTrainingAccuracy.config(text = "Accuracy: {:.5f}".format(accuracy))
        win.update()


xWinSize=1024
yWinSize=768
noSample=100
stepDelay=0.01
layerDelay=0
winFontSize=20
listFontSize=12
yLineUp=0.5
yCompare=0.6
tStop=0

tXSize=1000
tYSize=150
tDotSize=30
tXStep=100

tOffset=3
tLineWidth=5

tDObj=[]
tGuess=[]
tStrCfgCNNSize='10\n10'
tCfgCNNSize=[10,10]
tCfgCNN=[]
tCWidth=1024
tCHeight=200
tCXOffset=5
tCYOffset=5
tCObj=[]
tNoEpoch=100
tAccumulatedEpoch=0
tLearnRate=0.01
tCNN=[]
tCNNAccuracy=[]
tAObj=[]

tIWidth=1024
tIHeight=200
tIObj=[]

tAWidth=300
tAHeight=200

tRWidth=300
tRHeight=200
tRObj=[]


tObj=[]
tGameStat=[0,0,0,0]
mouseXY=[0,0]

win=tk.Tk()
win.title("Machine Learning Classification")
win.geometry("1024x768")

#lNumberImage= tk.Label(win, text = "Digit images to train", font=("Arial", winFontSize))
#lNumberImage.place(relx=0.02, rely=0.025, anchor=tk.W)

canvNumber=tk.Canvas(win, width=tXSize, height=tYSize)
canvNumber.place(relx=0.5, rely=0.05, anchor=tk.N)

#CNN network connection
canvC=tk.Canvas(win, width=tCWidth, height=tCHeight)
canvC.place(relx=0.50, rely=0.47, anchor=tk.N)
tttRectC=canvC.create_rectangle(tOffset,tOffset,tCWidth-tOffset,tCHeight-tOffset,outline="gray")

#CNN matrix image display
canvI=tk.Canvas(win, width=tIWidth, height=tIHeight)
canvI.place(relx=0.50, rely=0.73, anchor=tk.N)
tttRectI=canvI.create_rectangle(tOffset,tOffset,tIWidth-tOffset,tIHeight-tOffset,outline="gray")

#CNN accuracy display
canvA=tk.Canvas(win, width=tAWidth, height=tAHeight)
canvA.place(relx=0.3, rely=0.20, anchor=tk.NW)
tttRectA=canvA.create_rectangle(tOffset,tOffset,tAWidth-tOffset,tAHeight-tOffset,outline="gray")

#CNN random image display
canvR=tk.Canvas(win, width=tRWidth, height=tRHeight)
canvR.place(relx=0.7, rely=0.20, anchor=tk.NW)
tttRectA=canvR.create_rectangle(tOffset,tOffset,tRWidth-tOffset,tRHeight-tOffset,outline="gray")


#left column
textNetworkArch = tk.Label(win, text = "Machine Nodes at Layer", font=("Arial", winFontSize))
textNetworkArch.place(relx=0.02, rely=0.02, anchor=tk.W)

inputNetworkArch = tk.Text(win, height = 7, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNetworkArch.place(relx=0.02, rely=0.04, anchor=tk.NW)
inputNetworkArch.insert(tk.END, tStrCfgCNNSize)

bNetworkInitialize = tk.Button(win, text = "Initialize",  command = cCNNInit, font=("Arial", winFontSize))
bNetworkInitialize.place(relx=0.10, rely=0.07, anchor=tk.W)


#
textTrainData = tk.Label(win, text = "Training image data size:", font=("Arial", winFontSize))
textTrainData.place(relx=0.02, rely=0.27, anchor=tk.W)
#
textTrainDataSize = tk.Label(win, text = str(train_images.shape), font=("Arial", winFontSize))
textTrainDataSize.place(relx=0.02, rely=0.31, anchor=tk.W)

textTestData = tk.Label(win, text = "Test image data size:", font=("Arial", winFontSize))
textTestData.place(relx=0.02, rely=0.35, anchor=tk.W)
#
textTestDataSize = tk.Label(win, text = str(test_images.shape), font=("Arial", winFontSize))
textTestDataSize.place(relx=0.02, rely=0.39, anchor=tk.W)


#center column
textNoEpoch = tk.Label(win, text = "Training epochs", font=("Arial", winFontSize))
textNoEpoch.place(relx=0.30, rely=0.02, anchor=tk.W)

inputNoEpoch = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNoEpoch.place(relx=0.50, rely=0.02, anchor=tk.W)
inputNoEpoch.insert(tk.END, tNoEpoch)


#
textLearnRate = tk.Label(win, text = "Learning rate", font=("Arial", winFontSize))
textLearnRate.place(relx=0.30, rely=0.06, anchor=tk.W)

inputLearnRate= tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputLearnRate.place(relx=0.50, rely=0.06, anchor=tk.W)
inputLearnRate.insert(tk.END, tLearnRate)

#start train button
bTrain = tk.Button(win, text = "Start training",  command = cTrain, font=("Arial", winFontSize))
bTrain.place(relx=0.3, rely=0.10, anchor=tk.W)

bStop = tk.Button(win, text = "Stop",  command = cStop, font=("Arial", winFontSize))
bStop.place(relx=0.5, rely=0.10, anchor=tk.W)

textTrainingCount = tk.Label(win, text = "Epoch:", font=("Arial", winFontSize))
textTrainingCount.place(relx=0.30, rely=0.14, anchor=tk.W)
textTrainingAccuracy = tk.Label(win, text = "Accuracy:", font=("Arial", winFontSize))
textTrainingAccuracy.place(relx=0.30, rely=0.18, anchor=tk.W)

#start train button
bEvaluate = tk.Button(win, text = "Evaluate neural network",  command = cEvaluate, font=("Arial", winFontSize))
bEvaluate.place(relx=0.7, rely=0.02, anchor=tk.W)

textEvaluateAccuracy = tk.Label(win, text = "Accuracy:", font=("Arial", winFontSize))
textEvaluateAccuracy.place(relx=0.7, rely=0.06, anchor=tk.W)

bEvaluateOneImage = tk.Button(win, text = "Evaluate a random image",  command = cEvaluateRandom, font=("Arial", winFontSize))
bEvaluateOneImage.place(relx=0.7, rely=0.10, anchor=tk.W)

textEvaluateRandom = tk.Label(win, text = "Guess", font=("Arial", winFontSize))
textEvaluateRandom.place(relx=0.7, rely=0.14, anchor=tk.W)
textEvaluateRandomProb = tk.Label(win, text = "Probability:", font=("Arial", winFontSize))
textEvaluateRandomProb.place(relx=0.7, rely=0.18, anchor=tk.W)

win.mainloop()