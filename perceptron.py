#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 17:37:36 2022

@author: miguel
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math

def Coste(s_red, s_deseado):
    return np.mean((s_red - s_deseado)**2)

def DCoste(s_red, s_deseado):
    return (s_red - s_deseado)
 
class Perceptron():
    
  def __init__(self, y, salidaCapa):
    self.SalidaCapa = salidaCapa
    self.SumaPonderada = y
  
  @staticmethod
  def calcSalida(data, capa):
    y = data @ capa.W + capa.B
    return Perceptron(y, capa.f_activación(y))

class CapaRed():
    
  def __init__(self, entradas, salidas):
    self.B = (np.random.rand(1,salidas) * 2) - 1
    self.W = (np.random.rand(entradas, salidas) * 2) - 1

  def f_activación(self, x):
    return 1 / (1 + np.e**(-x))

  def d_f_activacion(self, x):
    return x * (1 - x)

  @staticmethod
  def setConfig(tamaños):
    capas=[]
    for i in range(len(tamaños)-1):
      capas.append(CapaRed(tamaños[i], tamaños[i+1]))
    return capas
 
  @staticmethod
  def Entrenar(capas,dataSet,labels,f_aprendizaje=0.5):
    
    # Forward Pass
    capasRed=CapaRed.calcSalida(capas, dataSet)
    
    # Backpropagation
    # Calculo delta ds - última capa
    deltas=[DCoste(capasRed[-1].SalidaCapa,labels)* capas[-1].d_f_activacion(capasRed[-1].SalidaCapa)]
    
    # Calculo deltas - demas capas
    for i in reversed(range(0,len(capas)-1)):
      deltas.insert(0, deltas[0] @ capas[i+1].W.T * capas[i].d_f_activacion(capasRed[i+1].SalidaCapa))
    
    # Ajuste con descenso del gradiente
    for i in reversed(range(0,len(capas))):
      capas[i].B=capas[i].B - np.mean(deltas[i],axis=0,keepdims=True) * f_aprendizaje
      capas[i].W=capas[i].W - capasRed[i].SalidaCapa.T @ deltas[i] * f_aprendizaje
    
    capaResultado=capasRed[-1]
    return capaResultado.SalidaCapa#la salida es el cálculo que ha hecho la red
 
  @staticmethod
  def calcSalida(capas, dataSet):
    capas_n=[Perceptron(None, dataSet)]
    for i in range(len(capas)):
      capas_n.append(Perceptron.calcSalida(capas_n[-1].SalidaCapa, capas[i]))
    return capas_n
 
  @staticmethod
  def classify(capas, dataSet):
    capaResultado=CapaRed.calcSalida(capas, dataSet)[-1]
    return capaResultado.SalidaCapa

def generar_puntos(n = 500, show = True):
    width = 2
        
    r = width/2
    
    x = np.random.random_sample((1,int(n/4)))
    x2 = np.random.random_sample((1,int(n/4)))
    x3 = np.random.random_sample((1,int(n/4)))
    x4 = np.random.random_sample((1,int(n/4)))
    x = np.concatenate((x,x2,-1*x3,-1*x4), axis=None)*width
    y = np.random.random_sample((1,int(n/4)))
    y2 = np.random.random_sample((1,int(n/4)))
    y3 = np.random.random_sample((1,int(n/4)))
    y4 = np.random.random_sample((1,int(n/4)))
    y = np.concatenate((y,-1*y2,y3,-1*y4), axis=None)*width
    
    attrib = np.vstack((x,y))
    
    #print(x.shape,y.shape, attrib.shape)
    
    labels = []
    afuera = []
    adentro = []
    att = []
    for i in range(len(attrib[0])):
        if (math.sqrt(attrib[0,i]**2 + attrib[1,i]**2) < r-(r*0.1)):
            labels.append([1])
            adentro.append(attrib[:,i])
            att.append(attrib[:,i])
        else:
            if (math.sqrt(attrib[0,i]**2 + attrib[1,i]**2) > r+(r*0.1) and 
                math.sqrt(attrib[0,i]**2 + attrib[1,i]**2) < width):
                labels.append([0])
                afuera.append(attrib[:,i])
                att.append(attrib[:,i])
            
    labels = np.array(labels)
    afuera = np.array(afuera).T
    adentro = np.array(adentro).T
    #print(afuera.shape, adentro.shape)
    if show:
        figure, axes = plt.subplots()
        axes = plt.scatter(afuera[0,:], afuera[1,:], color='red')
        axes = plt.scatter(adentro[0,:], adentro[1,:], color='blue')
        axes = plt.title("DataSet entrenamiento")
    #print(attrib.shape, labels.shape)
    return np.array(att), labels


attrib, labels = generar_puntos(1000)
attrib_testing, labels_testing = generar_puntos(1000, False)
#print(attrib.shape, labels.shape)

# nElementos=500#n
# nCaracteristicasElemento=2#p
# attrib, labels=MakeCircles(n_samples=nElementos,factor=0.5,noise=0.064)
# labels=labels[:,np.newaxis]

# attrib_testing, labels_testing=MakeCircles(n_samples=nElementos,factor=0.5,noise=0.064)
# labels_testing=labels[:,np.newaxis]

print(attrib.shape, labels.shape)
print(attrib_testing.shape, labels_testing.shape)


error=[]
epoch=0
max_epocas=10000
target=0.001

figure, axes = plt.subplots()

capas=CapaRed.setConfig([2,4,1])

while epoch<max_epocas and (len(error)==0 or error[-1]>target):
  Salida=CapaRed.Entrenar(capas, attrib, labels, 0.05)
  
  epoch += 1
  
  if (epoch % 25 == 0): 
      error.append(Coste(Salida, labels))
      res = CapaRed.classify(capas,attrib_testing)
      good, bad = 0, 0
      afuera = []
      adentro = []
      for i, r in enumerate(res):
        if r > 0.8:
            r = 1
        else:
            if r < 0.2:
                r = 0
            else:
                r = -1
            
        if (r == 1):
            adentro.append(attrib_testing[i,:])
        else:
            afuera.append(attrib_testing[i,:])        
            
        if (r == labels_testing[i]):
            good = good + 1
            #print("Esperado:", labels[i], " Obtenido:", res[0], "OK")
        else:
            bad = bad + 1
            #print("Esperado:", labels[i], " Obtenido:", res[0], "FALLÓ")
    
      #print(res)
      figure.clf()
      axes = plt.subplot(2,1,1)
      axes = plt.axis = "equal"
      if len(afuera) > 2:
        axes = plt.scatter(np.array(afuera).T[0,:], np.array(afuera).T[1,:], color='red')
      if len(adentro) > 2:
        axes = plt.scatter(np.array(adentro).T[0,:], np.array(adentro).T[1,:], color='blue')
      axes = plt.subplot(2,1,2)
      axes = plt.axis = "equal"
      axes = plt.plot(range(len(error)), error)
      plt.pause(0.001)  
      plt.show(block=False) 
      
      n_testing = len(labels)
      good_p = round((good/n_testing)*100,2)
      bad_p = round((bad/n_testing)*100,2)
      print("Good(",good,"): ", good_p, "%, Bad(",bad,"):", bad_p, "%, total: ", n_testing, "epocas: ", epoch, "/", max_epocas, " Error: ", error[-1], end='\r' )

plt.show()
  
exit()
#***************************************
verProgreso=50
res=50
circleOut=0
circleIn=1
colorCircleOut="skyblue"
colorCircleIn="salmon"
loss=[]
i=0
f=5000
umbralFin=0.001
capas=CapaRed.Crear([nCaracteristicasElemento,4,8,1])

while i<f and (len(loss)==0 or loss[-1]>umbralFin):
  Perceptron=CapaRed.Entrenar(capas,circulosDataSet,circulosIdAt,0.05)
  #print(Perceptron)
  #muestro el entrenamiento
  if i%verProgreso == 0:
    loss.append(Coste(Perceptron,circulosIdAt))
 
    _x0=np.linspace(-1.5,1.5,res)
    _x1=np.linspace(-1.5,1.5,res)
    _y=np.zeros((res,res))
    for i0,x0 in enumerate(_x0):
      for i1,x1 in enumerate(_x1):
        _y[i0,i1]=CapaRed.classify(capas,np.array([[x0,x1]]))[0][0]
 
    pyplot.pcolormesh(_x0,_x1,_y,cmap="coolwarm")
    pyplot.axis("equal")
 
    pyplot.title("Iteración "+str(i))
    pyplot.scatter(circulosDataSet[circulosIdAt[:,0]==circleOut,0],circulosDataSet[circulosIdAt[:,0]==circleOut,1],c=colorCircleOut)
    pyplot.scatter(circulosDataSet[circulosIdAt[:,0]==circleIn,0],circulosDataSet[circulosIdAt[:,0]==circleIn,1],c=colorCircleIn)  
 
    Clear(wait=True)
    pyplot.show()
    pyplot.plot(range(len(loss)),loss)
    pyplot.show()
  i=i+1
if i<f:
  print("Acabado en la iteración "+str(i))
else:
  print("Modelo sin acabar")

