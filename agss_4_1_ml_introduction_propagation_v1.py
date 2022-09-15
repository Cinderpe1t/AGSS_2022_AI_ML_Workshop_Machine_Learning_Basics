import time
import tkinter as tk
import random
import numpy as np
import math
import copy

n0=[[0, 1, 0],[1, 0, 1],[1, 0, 1],[1, 0, 1],[0, 1, 0]]
n1=[[0, 1, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0]]
n2=[[1, 1, 0],[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 1, 1]]
n3=[[1, 1, 0],[0, 0, 1],[0, 1, 1],[0, 0, 1],[1, 1, 0]]
n4=[[1, 0, 1],[1, 0, 1],[1, 1, 1],[0, 0, 1],[0, 0, 1]]
n5=[[1, 1, 1],[1, 0, 0],[1, 1, 1],[0, 0, 1],[1, 1, 0]]
n6=[[1, 1, 1],[1, 0, 0],[1, 1, 1],[1, 0, 1],[1, 1, 1]]
n7=[[1, 1, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[0, 1, 0]]
n8=[[1, 1, 1],[1, 0, 1],[1, 1, 1],[1, 0, 1],[1, 1, 1]]
n9=[[1, 1, 1],[1, 0, 1],[1, 1, 1],[0, 0, 1],[0, 0, 1]]

numberImage=[n3]
tCX=np.array(numberImage)
tCX=np.reshape(tCX,(1,15))
tCX=np.transpose(tCX)

tCY=np.eye(1)

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

def cAnimFastest():
    global stepDelay
    stepDelay=0
    inputSpeed.delete('1.0', tk.END)
    inputSpeed.insert(tk.END, stepDelay)
    
def cAnimationDelay():
    global stepDelay
    inp = inputSpeed.get(1.0, "end-1c")
    stepDelay = float(inp)

def cTrain():
    global tNoEpoch, tLearnRate
    #inp = inputNoEpoch.get(1.0, "end-1c")
    #tNoEpoch = int(inp)
    inp = inputSpeed.get(1.0, "end-1c")
    stepDelay = float(inp)
    inp = inputLearnRate.get(1.0, "end-1c")
    tLearnRate = float(inp)

    if len(tCfgCNN)==0:
        cCNNInit()        
    train()

def cCheck():
    global tObj, mouseXY, tCoord, tGameStat, tLastMove, tBoard, tNoGames, tP1LastMove, tP2LastMove
    tGameStat=[0,0,0,0]
    tPlayTurn=0
    
def cStop():
    global tStop
    tStop=1

def rgbToHex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

def drawCNN():
    global tCfgCNNSize, tCfgCNN, tCNN, tCObj
    #clear if there is existing drawing
    if len(tCObj)>0:
        for i in range(len(tCObj)):
            canvC.delete(tCObj[i])
    tXos=0.08*tCWidth
    tDX=0.84*tCWidth
    
    #Connections
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        tX1=k*tDX/(len(tCfgCNNSize)-1)+tXos
        tX2=(k+1)*tDX/(len(tCfgCNNSize)-1)+tXos
        for i in range(tCfgCNNSize[k]):
            tY1=(i+0.5)*tCHeight/(tCfgCNNSize[k])
            for j in range(tCfgCNNSize[k+1]):
                tY2=(j+0.5)*tCHeight/(tCfgCNNSize[k+1])
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
        tX=i*tDX/(len(tCfgCNNSize)-1)+tXos
        for j in range(tCfgCNNSize[i]):
            tY=(j+0.5)*tCHeight/(tCfgCNNSize[i])
            tO=canvC.create_oval(tX-5,tY-5,tX+5,tY+5, width = 1, outline='black')
            tCObj.append(tO)

def drawForwardNode(i, values):
    global tCfgCNNSize, tCfgCNN, tCNN, tCObj    
    tempObj=[]
    tXos=0.08*tCWidth
    tDX=0.84*tCWidth
    #Draw nodes
    tX=i*tDX/(len(tCfgCNNSize)-1)+tXos
    for j in range(tCfgCNNSize[i]):
        tValue=values[j][0]
        tValue=np.maximum(tValue,0)
        tValue=np.minimum(tValue,1)

        tY=(j+0.5)*tCHeight/(tCfgCNNSize[i])
        if values[j]>0.5:
            colorValue=int(255*(tValue-0.5)*2)
            strColor=rgbToHex(255,255-colorValue,255-colorValue)
        elif values[j]<0.5:
            colorValue=abs(int(255*(0.5-tValue)*2))
            strColor=rgbToHex(255-colorValue,255-colorValue,255)
        else:
            strColor=rgbToHex(255,255,255)
        
        tO=canvC.create_oval(tX-5,tY-5,tX+5,tY+5, width = 3, outline='black', fill = strColor)
        tempObj.append(tO)
    
    #draw image flow
    tempObjI=[]
    #section 1
    sArw=7
    tY1=0.5*tIHeight
    tY2=0.05*tIHeight
    tY3=0.10*tIHeight

    k=i
    tX1=(k+0.3)*tIWidth/(len(tCfgCNNSize))
    tX2=(k+0.4)*tIWidth/(len(tCfgCNNSize))
    tX3=(k+0.9)*tIWidth/(len(tCfgCNNSize))

    dX=(tX2-tX1)*0.8
    if k==0:
        tO=canvI.create_line(tX3,tY2,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX3-sArw,tY3-sArw,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX3+sArw,tY3-sArw,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
    elif k==len(tCfgCNNSize)-1:
        tO=canvI.create_line(tX1,tY1,tX2,tY1, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX2-sArw,tY1-sArw,tX2,tY1, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX2-sArw,tY1+sArw,tX2,tY1, width = 3, fill='black')
        tempObjI.append(tO)

    else:
        tO=canvI.create_line(tX1,tY1,tX2,tY1, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX2,tY1,tX2,tY2, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX2,tY2,tX3,tY2, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX3,tY2,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX3-sArw,tY3-sArw,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
        tO=canvI.create_line(tX3+sArw,tY3-sArw,tX3,tY3, width = 3, fill='black')
        tempObjI.append(tO)
    
    win.update()
    if stepDelay>0:
        time.sleep(stepDelay)
    #clear if there is existing drawing
    if len(tempObj)>0:
        for i in range(len(tempObj)):
            canvC.delete(tempObj[i])
    if len(tempObjI)>0:
        for i in range(len(tempObjI)):
            canvI.delete(tempObjI[i])
    win.update()

def drawBackwardWeight(idx, Weight):
    global tCfgCNNSize, tCfgCNN, tCNN, tCObj
    #clear if there is existing drawing
    tempObj=[]
    tXos=0.08*tCWidth
    tDX=0.84*tCWidth
    
    k=idx-1
    #Update Connections
    minValue=np.amin(Weight)
    maxValue=np.amax(Weight)
    tX1=k*tDX/(len(tCfgCNNSize)-1)+tXos
    tX2=(k+1)*tDX/(len(tCfgCNNSize)-1)+tXos
    for i in range(tCfgCNNSize[k]):
        tY1=(i+0.5)*tCHeight/(tCfgCNNSize[k])
        for j in range(tCfgCNNSize[k+1]):
            tY2=(j+0.5)*tCHeight/(tCfgCNNSize[k+1])
            if Weight[j][i]>0:
                colorValue=int(255*Weight[j][i]/maxValue)
                strColor=rgbToHex(255,255-colorValue,255-colorValue)
            elif Weight[j][i]<0:
                colorValue=abs(int(255*Weight[j][i]/minValue))
                strColor=rgbToHex(255-colorValue,255-colorValue,255)
            else:
                strColor=rgbToHex(255,255,255)
            tO=canvC.create_line(tX1,tY1,tX2,tY2, width = 3, fill=strColor)
            tempObj.append(tO)

    #Update image
    tempObjI=[]
    tX1=(k+0.5)*tIWidth/(len(tCfgCNNSize))
    tX2=(k+1.5)*tIWidth/(len(tCfgCNNSize))
    dX=(tX2-tX1)*0.8
    for i in range(tCfgCNNSize[k]):
        tY1=0.10*tIHeight
        tY2=0.90*tIHeight
        dY=tY2-tY1
        for j in range(tCfgCNNSize[k+1]):
            if Weight[j][i]>0:
                colorValue=int(255*Weight[j][i]/maxValue)
                strColor=rgbToHex(255,255-colorValue,255-colorValue)
            elif Weight[j][i]<0:
                colorValue=abs(int(255*Weight[j][i]/minValue))
                strColor=rgbToHex(255-colorValue,255-colorValue,255)
            else:
                strColor=rgbToHex(255,255,255)
            ttX1=tX1+i*dX/tCfgCNNSize[k]
            ttX2=tX1+(i+1)*dX/tCfgCNNSize[k]
            ttY1=tY1+j*dY/tCfgCNNSize[k+1]
            ttY2=tY1+(j+1)*dY/tCfgCNNSize[k+1]
            tO=canvI.create_rectangle(ttX1,ttY1,ttX2,ttY2,fill=strColor,outline='black',width = 3)
            tempObjI.append(tO)

    win.update()
    if stepDelay>0:
        time.sleep(stepDelay)
    #clear after display
    if len(tempObj)>0:
        for i in range(len(tempObj)):
            canvC.delete(tempObj[i])
    if len(tempObjI)>0:
        for i in range(len(tempObjI)):
            canvI.delete(tempObjI[i])

    win.update()

def displayNetworkImage():
    global tCfgCNNSize, tCfgCNN, tCNN, tIObj
    #delete previous objects
    if len(tIObj)>0:
        for i in range(len(tIObj)):
            canvI.delete(tIObj[i])
    #Connections
    for k in range(len(tCfgCNNSize)-1):
        tW=tCNN["W" + str(k+1)]
        minValue=np.amin(tW)
        maxValue=np.amax(tW)
        #print(tW)
        tX1=(k+0.5)*tIWidth/(len(tCfgCNNSize))
        tX2=(k+1.5)*tIWidth/(len(tCfgCNNSize))
        dX=(tX2-tX1)*0.8
        for i in range(tCfgCNNSize[k]):
            tY1=0.10*tIHeight
            tY2=0.90*tIHeight
            dY=tY2-tY1
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

def cCNNInit():
    global tCfgCNNSize, tCfgCNN, tCNN, tGuess, tCNNAccuracy
    inputString = inputNetworkArch.get(1.0, "end-1c")
    tCfgCNNSize=[15] #size of image
    tCNNAccuracy=[]
    tGuess=[]
    
    ListString=inputString.split('\n')
    for i in range(len(ListString)):
        if len(ListString[i])>0:
            tCfgCNNSize.append(int(ListString[i]))
    tCfgCNNSize.append(1) #last output

    tCfgCNN=[]
    for i in range(len(tCfgCNNSize)-1):
        if i<len(tCfgCNNSize)-2:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "relu"})
        else:
            tCfgCNN.append({"input_dim": tCfgCNNSize[i], "output_dim": tCfgCNNSize[i+1], "activation": "sigmoid"})

    #print(tCfgCNN)
    tCNN=init_layers(tCfgCNN, seed = 99)
    
    #display CNN
    guessDigitPlot()
    drawCNN()
    displayNetworkImage()
    drawAccuracy()
    win.update()
            
def guessDigitPlot():
    global tGuess, tDObj
    
    #delete previous objects
    if len(tDObj)>0:
        for i in range(len(tDObj)):
            canvNumber.delete(tDObj[i])
            
    for k in range(len(numberImage)):
        tNumImage=numberImage[k]
        #print(tNumImage)
        tXAnchor=k*tXStep+tOffset
        if len(tGuess)==0:
            strColor="gray"
        elif tGuess[k]==True:
            strColor="black"
        else:
            strColor="gray"
        for i in range(5):
            tY=i*tDotSize
            for j in range(3):
                if tNumImage[i][j]==1:
                    tX=tXAnchor+j*tDotSize
                    #print(tX,tY)
                    tO=canvNumber.create_rectangle(tX,tY,tX+tDotSize,tY+tDotSize,fill=strColor)
                    tDObj.append(tO)
                
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
        #Highlight nodes
        drawForwardNode(idx,A_curr)
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

    #Highlight last node
    drawForwardNode(idx+1,A_curr)
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
    #for layer_idx, layer in enumerate(nn_architecture, 1):
    for layer_idx, layer in reversed(list(enumerate(nn_architecture, 1))):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]        
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]
        drawBackwardWeight(layer_idx, params_values["W" + str(layer_idx)])
    return params_values;

def train():
    global tNoEpoch, tLearnRate, tCNN, tCNNAccuracy, tStop, tAccumulatedEpoch
    #numpy cnn code from https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
    
    # performing calculations for subsequent iterations
    for i in range(tNoEpoch):
        # step forward
        Y_hat, cashe = full_forward_propagation(tCX, tCNN, tCfgCNN)
        #print(Y_hat)
        
        
        # calculating metrics and saving them in history
        #cost = get_cost_value(Y_hat, Y)
        #cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, tCY)
        tCNNAccuracy.append(Y_hat[0][0])
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, tCY, cashe, tCNN, tCfgCNN)
        # updating model state
        tCNN = update(tCNN, grads_values, tCfgCNN, tLearnRate)
        #if stepDelay>0:
        #    time.sleep(stepDelay)
        if(i % 10 == 0):
            win.update()
            #print("Iteration: {:05} - accuracy: {:.5f}".format(i, accuracy))
        drawCNN()
        drawAccuracy()
        displayNetworkImage()
        guessDigitPlot()
        win.update()

        if tStop==1:
            tStop=0
            textTrainingCount.config(text = "Epoch: %4d / %4d / %4d - stopped" % (i+1,tNoEpoch,tAccumulatedEpoch))
            return
        tAccumulatedEpoch+=1
        textTrainingCount.config(text = "Epoch: %4d / %4d / %4d" % (i+1,tNoEpoch,tAccumulatedEpoch))
        textTrainingAccuracy.config(text = "Accuracy: {:.3f}".format(Y_hat[0][0]))            
    #return params_value

def drawAccuracy():
    global tCNNAccuracy, tAObj, tAHeight, tOffset, tAWidth
    #clear if there is existing drawing
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
    if len(tCNNAccuracy)<100:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/100
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff*2)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            
    else:
        for k in range(len(tCNNAccuracy)):
            tX=tOff+k*(tAWidth-tOff*2)/len(tCNNAccuracy)
            tY=tOff+(1-tCNNAccuracy[k])*(tAHeight-tOff*2)
            tO=canvA.create_oval(tX-1,tY-1,tX+1,tY+1, width = 1, outline='red')
            tAObj.append(tO)            

xWinSize=1024
yWinSize=768
noSample=100
stepDelay=2
layerDelay=0
winFontSize=20
listFontSize=12
yLineUp=0.5
yCompare=0.6
tStop=0

tXSize=100
tYSize=150
tDotSize=30
tXStep=100

tOffset=3
tLineWidth=5

tDObj=[]
tGuess=[]
tStrCfgCNNSize='10\n5'
tCfgCNNSize=[10,5]
tCfgCNN=[]
tCWidth=1000
tCHeight=270
tCXOffset=5
tCYOffset=5
tCObj=[]
tNoEpoch=100
tAccumulatedEpoch=0
tLearnRate=0.01
tCNN=[]
tCNNAccuracy=[]
tAObj=[]

tIWidth=700
tIHeight=280
tIObj=[]

tAWidth=280
tAHeight=280

tObj=[]
tGameStat=[0,0,0,0]
mouseXY=[0,0]

win=tk.Tk()
win.title("Machine Learning Introduction")
win.geometry("1024x768")

lNumberImage= tk.Label(win, text = "Image to train", font=("Arial", winFontSize))
lNumberImage.place(relx=0.02, rely=0.02, anchor=tk.W)

canvNumber=tk.Canvas(win, width=tXSize, height=tYSize)
canvNumber.place(relx=0.02, rely=0.04, anchor=tk.NW)

#draw digits
guessDigitPlot()

#CNN network connection
canvC=tk.Canvas(win, width=tCWidth, height=tCHeight)
canvC.place(relx=0.5, rely=0.25, anchor=tk.N)
tttRectC=canvC.create_rectangle(tOffset,tOffset,tCWidth-tOffset,tCHeight-tOffset,outline="gray")

#CNN matrix image display
canvI=tk.Canvas(win, width=tIWidth, height=tIHeight)
canvI.place(relx=0.35, rely=0.60, anchor=tk.N)
tttRectI=canvI.create_rectangle(tOffset,tOffset,tIWidth-tOffset,tIHeight-tOffset,outline="gray")

#CNN accuracy display
canvA=tk.Canvas(win, width=tAWidth, height=tAHeight)
canvA.place(relx=0.85, rely=0.60, anchor=tk.N)
tttRectA=canvA.create_rectangle(tOffset,tOffset,tAWidth-tOffset,tAHeight-tOffset,outline="gray")


#left column
textNetworkArch = tk.Label(win, text = "# of Nodes at Layer", font=("Arial", winFontSize))
textNetworkArch.place(relx=0.20, rely=0.02, anchor=tk.W)

inputNetworkArch = tk.Text(win, height = 5, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputNetworkArch.place(relx=0.20, rely=0.04, anchor=tk.NW)
inputNetworkArch.insert(tk.END, tStrCfgCNNSize)

bNetworkInitialize = tk.Button(win, text = "Initialize",  command = cCNNInit, font=("Arial", winFontSize))
bNetworkInitialize.place(relx=0.28, rely=0.06, anchor=tk.W)

#center column
#
#textLayerSpeed = tk.Label(win, text = "Layer delay (sec)", font=("Arial", winFontSize))
#textLayerSpeed.place(relx=0.30, rely=0.30, anchor=tk.W)

#inputLayerSpeed = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
#inputLayerSpeed.place(relx=0.50, rely=0.30, anchor=tk.W)
#inputLayerSpeed.insert(tk.END, layerDelay)

#
textSpeed = tk.Label(win, text = "Play delay (sec)", font=("Arial", winFontSize))
textSpeed.place(relx=0.45, rely=0.02, anchor=tk.W)

inputSpeed = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputSpeed.place(relx=0.60, rely=0.02, anchor=tk.W)
inputSpeed.insert(tk.END, stepDelay)

bFaster = tk.Button(win, text = "Faster",  command = cAnimFaster, font=("Arial", winFontSize))
bFaster.place(relx=0.70, rely=0.02, anchor=tk.W)

bSlower = tk.Button(win, text = "Slower",  command = cAnimSlower, font=("Arial", winFontSize))
bSlower.place(relx=0.80, rely=0.02, anchor=tk.W)

bStop = tk.Button(win, text = "Fastest",  command = cAnimFastest, font=("Arial", winFontSize))
bStop.place(relx=0.90, rely=0.02, anchor=tk.W)

#
#textNoEpoch = tk.Label(win, text = "Training epochs", font=("Arial", winFontSize))
#textNoEpoch.place(relx=0.30, rely=0.30, anchor=tk.W)

#inputNoEpoch = tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
#inputNoEpoch.place(relx=0.50, rely=0.30, anchor=tk.W)
#inputNoEpoch.insert(tk.END, tNoEpoch)

#
textLearnRate = tk.Label(win, text = "Learning rate", font=("Arial", winFontSize))
textLearnRate.place(relx=0.45, rely=0.06, anchor=tk.W)

inputLearnRate= tk.Text(win, height = 1, width = 5, bg = "light yellow", font=("Arial", winFontSize))
inputLearnRate.place(relx=0.60, rely=0.06, anchor=tk.W)
inputLearnRate.insert(tk.END, tLearnRate)

#start train button
bTrain = tk.Button(win, text = "Start training",  command = cTrain, font=("Arial", winFontSize))
bTrain.place(relx=0.45, rely=0.10, anchor=tk.W)

#start match button
#bPlay = tk.Button(win, text = "Evaluate",  command = cCheck, font=("Arial", winFontSize))
#bPlay.place(relx=0.5, rely=0.45, anchor=tk.W)

bStop = tk.Button(win, text = "Stop",  command = cStop, font=("Arial", winFontSize))
bStop.place(relx=0.60, rely=0.10, anchor=tk.W)

textTrainingCount = tk.Label(win, text = "Epoch:", font=("Arial", winFontSize))
textTrainingCount.place(relx=0.45, rely=0.14, anchor=tk.W)
textTrainingAccuracy = tk.Label(win, text = "Accuracy:", font=("Arial", winFontSize))
textTrainingAccuracy.place(relx=0.45, rely=0.18, anchor=tk.W)



win.mainloop()