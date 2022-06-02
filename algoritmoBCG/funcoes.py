import time

import numpy as np
import math


#                                                            _           __                  /\/|       _
#     /\                                                    | |         / _|                |/\/       | |
#    /  \   _ __ _ __ _   _ _ __ ___   __ _ _ __    ___  ___| |_ __ _  | |_ _   _ _ __   ___ __ _  ___ | |
#   / /\ \ | '__| '__| | | | '_ ` _ \ / _` | '__|  / _ \/ __| __/ _` | |  _| | | | '_ \ / __/ _` |/ _ \| |
#  / ____ \| |  | |  | |_| | | | | | | (_| | |    |  __/\__ \ || (_| | | | | |_| | | | | (_| (_| | (_) |_|
# /_/    \_\_|  |_|   \__,_|_| |_| |_|\__,_|_|     \___||___/\__\__,_| |_|  \__,_|_| |_|\___\__,_|\___/(_)
#                                                                                        )_)

def funcaoAptidaoBCG(alfa,beta,acuracia,numeroAtributos):
    return alfa*acuracia #+ 1*(1/numeroAtributos)

#[-100,100] -> f(0) = 2 # adaptei a função
def rastrigin1dim(x):
    funcao = 7 + x**2 - 5*np.cos(2*x*np.pi)
    return funcao

def esfera1dim(x):
    funcao = x**2
    return funcao

#[-1000,1000] -> f(100) = -200 adaptado
def bukin1dim(x):
    funcao = 20*np.sqrt(np.abs(x - (0.01*(x**2) ) )) - 2*abs(x)
    return funcao

#[-1000,1000] -> f(-1) = 0 adaptado
def levi1dim(x):
    funcao = np.power(np.sin(3*x*np.pi),2) + np.power(x+1,2) * (1+np.power(np.sin(3*x),2)) + np.power(x+1,2) * (1 + np.power(np.sin(x),2))
    return funcao


#[-1000,1000] -> f(10) = -20 adaptado
def mccormick1dim(x):
    funcao = np.sin(x*8) + np.power( (0.1+x**3) - (10*x),2) -2*x + 1
    return funcao

#[-1000,1000]  -> f(0) = 10
def Dig1dim(x):
    funcao = 10 + (np.power(5*np.sin(x),2)) + 0.04*x**2
    return funcao

# [-5.12,5.12]  -> f(0,0,0,..)  = 0
def rastriginCompleta(ninho, dimensoes):
    #print("TESTE",( np.power(0,2) - 10*np.cos(2*np.pi*0) ))
    #print("ninho[0]", ninho[0])
    #print("ninho[1]", ninho[1])
    #time.sleep(3)
    funcao = 10 * dimensoes
    for i in range(0, dimensoes):
        funcao = funcao + ( np.power(ninho[i],2) - 10*np.cos(2*np.pi*ninho[i]) )
    return funcao

#-512 a 512  -> f(512,404.2319) = -959.6407
def eggholder(ninho, dimensoes):
    aptidao = - (ninho[1] + 47) * np.sin(np.sqrt(abs(ninho[1] + (ninho[0]/2) +47))) - ninho[0] *np.sin(np.sqrt(abs(ninho[0] - (ninho[1]+47))))
    return aptidao

# [1000, 1000] -> f(0,0) = 0
def sphere(ninho, dimensoes):
    aptidao = 0
    for i in range(dimensoes):
        aptidao = aptidao + ninho[i]**2
    return aptidao

# [-5,5] -> f(0,0) = 0
def ackley(ninho, dimensoes):
    aptidao = -20*math.exp(-0.2*np.sqrt(0.5*(np.power(ninho[0],2)+np.power(ninho[1],2)) )) \
              - math.exp(0.5*(np.cos(2*math.pi*ninho[0])+np.cos(2*math.pi*ninho[1])) ) + math.e + 20

    return aptidao

# [-1000,1000] -> f(1,1) = 0 -> Função difícil
def rosenbrock(ninho, dimensoes):
    #print("ver:",np.power((1-1),2) + 100*np.power((1 - np.power(1,2)),2))
    aptidao = np.power((1-ninho[0]),2) + 100*np.power((ninho[1] - np.power(ninho[0],2)),2)
    return aptidao

# [-4.5,4.5] -> f(3,0.5) = 0
def beale(ninho,dimensoes):
    aptidao = np.power(1.5-ninho[0]+ninho[0]*ninho[1],2) \
              + np.power(2.25-ninho[0]+ninho[0]*np.power(ninho[1],2),2) \
              + np.power(2.625-ninho[0]+ninho[0]*np.power(ninho[1],3),2)
    return aptidao

# [-100,100] -> f(PI,PI) = -1
def easom(ninho,dimensoes):
    aptidao = -np.cos(ninho[0])*np.cos(ninho[1])*np.exp( -(np.power(ninho[0]-np.pi,2)+np.power(ninho[1]-np.pi,2)))
    return aptidao

# [-5,5] -> f(3,2) = 0; f(-2.805,3.1313) = 0; f(-3.779,-3.2831) = 0; f(3.5844,-1.8481) = 0
def himmelblau(ninho, dimensoes):
    aptidao = np.power(( np.power(ninho[0],2)+ninho[1]-11 ),2) + np.power(ninho[0]+np.power(ninho[1],2) - 7,2)
    return aptidao