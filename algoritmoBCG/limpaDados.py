#Lê um CSV e salva um CSV chamado file.csv com os dados limpos

import time
import pandas as pd

import palavrasParada
import preProcessador
import numpy
import sys
import textblob
from googletrans import Translator
from textblob import TextBlob

# generate random integer values
from random import seed
from random import randint

def main():

    print("Teste com SKLEARN")

    amostras = 200
    ativaPOStag = False

    print("\nDados brutos -", amostras, "amostras: \n")
    # dataset_imdb.csv comentarios_CLASME.csv comentarios_CLASME_2classes.csv opcovid_br_en.csv kaggle_sanders_10k.csv base_bruta_2_classes_ingles
    # Base completa: base_bruta_3_classes_cheia_ingles -> 12000 tuplas
    # Base menor: base_bruta_3_classes_ingles -> 6000 tuplas
    # Base rápida: base_rapida_ingles_2_classes -> 200, base_rapida_ingles_3_classes -> 300
    # Base que provavelmente será a final: base_bruta_ingles_3_classes_NOVA -> 6000
    # Base para quali base_bruta_ingles_3_classes_NOVA_1200 -> 1200
    dadosFilmes = pd.read_csv('comentarios/base_bruta_3_classes_cheia_ingles.csv')#.sample(amostras,random_state=1)##random_state=estadoInicial
    print(dadosFilmes.sentiment.value_counts())
    print(dadosFilmes.head())

    #Lista de dados textuais limpos
    listaFrasesLimpa = []

    #Lista de contextos em ordem de leitura
    listaContextos = dadosFilmes['context']

    indice = 0
    for sentenca in dadosFilmes.review:
        print("ANTES:",sentenca)
        #indice = dadosFilmes[dadosFilmes['review']==sentenca].index.item()
        print("indice:",indice)
        contextoLinha = listaContextos[indice]
        print("contexto: ", contextoLinha)
        token = preProcessador.preprocessadorParaSalvar(sentenca,contextoLinha,ativaPOStag)
        print("DEPOIS:",token)
        listaFrasesLimpa.append(sentenca)
        dadosFilmes.at[indice,'review'] = token
        print("####\n ")
        indice = indice + 1 #fecha o for
        #time.sleep(1)
    print("FIM")
    dadosFilmes.to_csv('comentarios/file.csv')

if __name__ == '__main__':
    main()