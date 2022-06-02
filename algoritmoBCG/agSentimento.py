import time
import random
import funcoes
import numpy as np
import math

import pandas as pd
import scipy
from cffi.backend_ctypes import xrange

import manipulacaoTexto
import palavrasParada
import preProcessador
import numpy
import sys

ponto = [0,0]

class buscaCucoGenetico():

    def __init__(self,limiteInferior,limiteSuperior,dimensoes,restoAtributos,tamanhoPop,Pa,alfa,beta,pc,pm,maxGeracoes,geracoesEstagnadas,dadosBrutos,matrizTexto,vetorTFIDF):
        #Fazem parte do carregamento inicial dos dados, não será alterado
        self.dadosBrutos = dadosBrutos
        self.matrizTexto = matrizTexto
        self.vetorTFIDF = vetorTFIDF

        #Definem o espaço de busca
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior

        self.dimensoes = dimensoes #X, Y ...
        self.restoAtributos = restoAtributos
        ##self.tamanhoGene = tamanhoGene # Representação dos reais em binário
        self.tamanhoPop = tamanhoPop # Deve ficar entre 20 e 100

        # calcula o número de bits dos limites no formato binário SEM O SINAL!!!
        qtd_bits_x_min = len(bin(limiteInferior).replace('0b', '' if limiteInferior < 0 else '+')) - 1
        qtd_bits_x_max = len(bin(limiteSuperior).replace('0b', '' if limiteSuperior < 0 else '+')) - 1

        # Pode ser obtido assim ou por parâmetro
        self.tamanhoGene = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min
        print("Tamanho GENE:", self.tamanhoGene)
        self.Pa = Pa
        self.alfa = alfa # Geralmente a=1
        self.beta = beta # Geralmente 0<=b<=2, aqui b=2

        self.pc = pc # Taxa de cruzamento
        self.pm = pm # Taxa de mutação

        self.maxGeracoes = maxGeracoes # De mil até 100k
        self.geracoesEstagnadas = geracoesEstagnadas # Talvez 100 até 10% de MaxGeracoes

        self._iniciaPopulacao()

    # OK
    # Recebe um ninho e retorna apenas os valores de string, isto é, o
    # vetor gene para montar a matrizTexto com atributos selecionados
    def retornaVetorGene(self, ninho):
        # vetorGene= [[] for i in range(self.tamanhoGene*self.dimensoes) ]
        vetorGene = []
        # print("Ninho no início:",ninho)
        indiceGenotipo = self.tamanhoGene
        indiceInicioGene = 0
        indiceFimGene = self.tamanhoGene - 1
        for dimensao in range(0, self.dimensoes):
            for indice in range(indiceInicioGene, indiceFimGene + 1):#Laço de 10 iterações
                vetorGene.append(int(ninho[indice]))
            # vetorGene[indiceInicioGene:indiceFimGene] = ninho[indiceInicioGene:indiceFimGene]
            indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
            indiceFimGene = indiceFimGene + self.tamanhoGene + 1
            indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza os indices
        # Para os atributos que sobraram, selecione todos!
        for dimensao in range(self.dimensoes, self.dimensoes+self.restoAtributos):
            vetorGene.append(1)
        #print("Vetor GENE do final:", vetorGene)

        return vetorGene

    def _iniciaPopulacao(self):
        # Inicia a população de ninhos
        self.populacao = [[] for i in range(self.tamanhoPop)]
        # Usado para controlar o AG
        self.populacaoAux = [[] for i in range(self.tamanhoPop)]

        #Vamos considerar que cada ninho tenha exatamente dois ovos -> x e y que compoem f(x,y) = Z
        for ninho in self.populacao:
            #Representa cada ovo
            for dim in range(0,self.dimensoes):
                XN = random.randint(self.limiteInferior, self.limiteSuperior)  # gera um número aleatório
                #print("Número aleatório gerado:",XN)

                num_bin = bin(XN).replace('0b', '' if XN < 0 else '+').replace('+','').zfill(self.tamanhoGene) # Converte para binário
                #print("num_bin",num_bin)

                for bit in num_bin:
                    ninho.append(bit)
                ninho.append(XN)
            #print("Tamanho do gene",self.tamanhoGene)
            #print("Ninho sem aptidão calculada:", ninho)
            vetorGene = self.retornaVetorGene(ninho)
            #print("Vetor gene:",vetorGene)

            aptidao = calculaAptidao(vetorGene,self.dadosBrutos,self.matrizTexto,self.vetorTFIDF)

            ninho.append(aptidao)
            #print("Ninho com aptidão calculada:", ninho)

            #print("Tamanho do ninho:",len(ninho))
            ultimaPosicao = len(ninho)
            #time.sleep(100)

        #ORDENAÇÃO
        #A posição dimensoes do ninho (última) sempre é a aptidão
        self.populacao = sorted(self.populacao, key=lambda ninho: ninho[ultimaPosicao-1])#Ordenar de acordo com a qualidade, em ordem crescente
        self.melhorIndividuoDoAG = self.populacao[self.tamanhoPop-1]
        print("melhorIndividuoDoAG na inicia_populacao:",self.melhorIndividuoDoAG)


    # Problemas de maximização - x é o individuo
    def encontraMaior(self, x1, x2, x3):
        if (x1[len(x1) - 1] > x2[len(x2) - 1]) and (x1[len(x1) - 1] > x3[len(x3) - 1]):
            return x1
        elif (x2[len(x2) - 1] > x3[len(x3) - 1]) and (x2[len(x2) - 1] > x1[len(x1) - 1]):
            return x2
        else:
            return x3

    # Problemas de minimização - x é o individuo
    def encontraMenor(self, x1, x2, x3):
        if (x1[len(x1)-1] < x2[len(x2)-1]) and (x1[len(x1)-1] < x3[len(x3)-1]):
            return x1
        elif (x2[len(x2)-1] < x3[len(x3)-1]) and (x2[len(x2)-1] < x1[len(x1)-1]):
            return x2
        else:
            return x3

    # Realiza a seleção do individuo mais apto por torneio, considerando N = 3
    def selecao(self):
        #tempoExecucao = time.time()
        self.melhorIndividuoDoAG = self.populacao[self.tamanhoPop - 1]#Salvar o melhor ninho para garantir o elitismo
        print("melhorIndividuoDoAG na SELEÇÃO:", self.melhorIndividuoDoAG)
        # Limpar a aux
        self.populacaoAux.clear()
        # Selecionar 3 indivíduos, até preencher uma nova população
        for j in range(0, self.tamanhoPop):
            individuo_1 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_2 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            individuo_3 = self.populacao[random.randint(0, self.tamanhoPop - 1)]
            #print("Ninho selecionado 1", individuo_1)
            #print("Ninho selecionado 2", individuo_2)
            #print("Ninho selecionado 3", individuo_3)

            individuoSelecionado = self.encontraMaior(individuo_1, individuo_2, individuo_3)
            self.populacaoAux.append(individuoSelecionado)
        #FECHAR FOR
        #print("Seleção:",self.populacaoAux)
        #time.sleep(10)
        #tempoExecucao = time.time() - tempoExecucao
        #print("Tempo da função Seleção AG:", tempoExecucao)
        # Agora a população auxiliar precisa ser cruzada

    # Operação de cross-over do AG
    def cruzamento(self):
        tempoExecucao = time.time()
        self.populacao.clear()  # Limpo para começar a preencher de novo
        i = 0  # Controle de index
        # A cada par de indivíduos, é necessário fazer a troca de genes, logo percorremos os pares até a metade da população
        for individuo in self.populacaoAux:
            #print("Indivíduo a ser cruzado:", individuo)
            # caso o crossover seja aplicado os pais cruzam seus genes e com isso geram dois filhos
            if i % 2 == 0:
                pai = individuo
            else:
                mae = individuo

            # Quando for uma geração ímpar, é pq é possível ter filhos e colocar na população original
            if i % 2 != 0:
                # Se for dentro do range
                if random.randint(1, 100) <= self.pc:  # 70%
                    indiceGenotipo = self.tamanhoGene
                    indiceInicioGene = 0
                    indiceFimGene = self.tamanhoGene - 1
                    parteIndividuo_1 = []
                    parteIndividuo_2 = []
                    filho_1 = []
                    filho_2 = []
                    # Para cada dimensão, eu preciso fazer um corte nos genes e recombinar um par de indivíduos
                    for dimensao in range(0, self.dimensoes):
                        ponto_de_corte = random.randint(indiceInicioGene + 1, indiceFimGene - 1)  # -2

                        #print("PONTO DE CORTE:", ponto_de_corte)
                        #print("INDEX PAI:", pai[indiceInicioGene:ponto_de_corte])  # Pega a parte esquerda do pai  [***|] do index para trás
                        #print("INDEX MAE:", mae[ponto_de_corte:indiceFimGene + 2])  # Pega a parte direita da mae  [|****] maior que index pra frente

                        parteIndividuo_1.append(pai[indiceInicioGene:ponto_de_corte] + mae[ponto_de_corte:indiceFimGene + 2])
                        parteIndividuo_2.append(mae[indiceInicioGene:ponto_de_corte] + pai[ponto_de_corte:indiceFimGene + 2])

                        indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                        indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                        indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza os indices

                    # Converte os filhos em array único e para lista para ficar igual antes
                    filho_1 = np.asarray(parteIndividuo_1).flatten().tolist()
                    filho_2 = np.asarray(parteIndividuo_2).flatten().tolist()

                    #print("FILHO A", filho_1)
                    #print("FILHO B", filho_2)

                    self.populacao.append(filho_1)
                    self.populacao.append(filho_2)
                else:
                    # caso contrário os filhos são cópias exatas dos pais
                    # É preciso dar um pop para evitar problemas na próxima geração
                    if (len(pai)) > ((self.dimensoes*self.tamanhoGene)+self.dimensoes):
                        #print("PAI:",pai)
                        #print("Tamanho pai:",len(pai))
                        #time.sleep(3)
                        while (len(pai) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes )):
                            pai.pop()
                    if (len(mae)) > ((self.dimensoes * self.tamanhoGene) + self.dimensoes):
                        #print("MAE:", mae)
                        #print("Tamanho mae:", len(mae))
                        while (len(mae) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes)):
                            mae.pop()

                    self.populacao.append(pai)
                    self.populacao.append(mae)

            i = i + 1  ##Fecha o for, controle de par e ímpar
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo da função Cruzamento AG:", tempoExecucao)

    #Operação de mutação do AG
    def mutacao(self):
        """
            Realiza a mutação dos bits de um indiviuo conforme uma dada probabilidade
            (taxa de mutação pm) e os coloca na população AUXILIAR
        """
        tempoExecucao = time.time()
        self.populacaoAux.clear()
        for individuo in self.populacao:  # cada indivíduo testa a sorte de ser mutado
            if random.randint(1, 100) <= self.pm:
                # quantidade = random.randint(1, int(self.tamanhoGene/2))  # Quantidade de bits a serem mutados
                # vetor = random.getrandbits(quantidade)
                # print("VETOR:", vetor)
                indiceGenotipo = self.tamanhoGene
                indiceInicioGene = 0
                indiceFimGene = self.tamanhoGene - 1
                for dimensoes in range(0,self.dimensoes):
                    for i in range(indiceInicioGene+1, indiceFimGene+1):#Mais 1 para evitar mudar o sinal
                        if random.randint(1, 10) >6: # Com isto, até 40% dos genes podem sofrer mutação -> x<=40%
                            if individuo[i] == '1':
                                individuo[i] = '0'
                            elif individuo[i] == '0':
                                individuo[i] = '1'
                            elif individuo[i] == '+':
                                individuo[i] = '-'
                            elif individuo[i] == '-':
                                individuo[i] = '+'
                    indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                    indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                    indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1
            #print("Indivíduo mutado:",individuo)
            #time.sleep(1)
            while (len(individuo) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes)):
                individuo.pop()
                print("Erro da mutação")
                time.sleep(100)
            self.populacaoAux.append(individuo)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo da função Mutação AG:", tempoExecucao)


    ##FUNÇÃO QUE CALCULA A APTIDAO E FAZ O CONTROLE DA POPULAÇÃO, RECEBE A PopAUX E COLOCA NA ORIGINAL
    #No caso do AG
    def avaliarAG(self):
        """
            Avalia as souluções produzidas, associando uma nota/avalição a cada elemento da população
            Lê a população auxiliar e os coloca na população original
        """
        tempoExecucaoAvaliarAG = time.time()
        self.populacao.clear()#Limpa inicialmente
        # Calcula o genótipo e fenótipo, fazer apenas aqui para evitar calculos desnecessários

        for ninho in self.populacaoAux:
            #print("NINHO_AUX",ninho)
            #print("TAMANHO_AUX",len(ninho))
            if len(ninho) > ( (self.tamanhoGene*self.dimensoes) + self.dimensoes ):
                print("Algo de errado!")
                print("NINHO_AUX", ninho)
                print("TAMANHO_AUX", len(ninho))
                while ( len(ninho) > ( (self.tamanhoGene*self.dimensoes) + self.dimensoes ) ):
                    ninho.pop()
                print("NINHO_AUX", ninho)
                print("TAMANHO_AUX", len(ninho))
                time.sleep(10)

        for ninho in self.populacaoAux:

            # inicialização dos indices
            indiceGenotipo = self.tamanhoGene
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1

            for dimensoes in range(0, self.dimensoes):
                # Na posição do genótipo, colocamos o genótipo
                ninho[indiceGenotipo] = int(calculaGenotipo(self.tamanhoGene, ninho, indiceInicioGene, indiceFimGene))

                # ninho[indiceGenotipo] = int(ninho[indiceGenotipo])
                if ninho[indiceGenotipo] > self.limiteSuperior:
                    #ninho[indiceGenotipo] = self.limiteSuperior
                    numero_bin = bin(self.limiteSuperior).replace('0b', '' if self.limiteSuperior < 0 else '+').replace('+','').zfill(self.tamanhoGene)  # Converte para binário
                    i = 0
                    for k in range(indiceInicioGene,indiceFimGene+1):
                        #print("numero_bin[i]:", numero_bin[i])
                        ninho[k] = numero_bin[i]
                        i = i + 1
                    ninho[indiceGenotipo] = self.limiteSuperior

                elif ninho[indiceGenotipo] < self.limiteInferior:
                    #ninho[indiceGenotipo] = self.limiteInferior
                    numero_bin = bin(self.limiteInferior).replace('0b', '' if self.limiteInferior < 0 else '+').replace('+','').zfill(self.tamanhoGene)  # Converte para binário
                    i = 0
                    for k in range(indiceInicioGene, indiceFimGene + 1):
                        #print("numero_bin[i]:", numero_bin[i])
                        ninho[k] = numero_bin[i]
                        i = i + 1
                    ninho[indiceGenotipo] = self.limiteSuperior

                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1

            # print("Ninho após genotipo:",ninho)
            # time.sleep(4)

            # Ultima posicao guarda a aptidão
            vetorGene = self.retornaVetorGene(ninho)
            # print("Vetor gene:",vetorGene)

            aptidao = calculaAptidao(vetorGene, self.dadosBrutos, self.matrizTexto,self.vetorTFIDF)
            ninho.append(aptidao)


            # Garante que o ninho tenha o tamanho certo
            if (len(ninho) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes + 1)):
            #    print("ninho wrong:", ninho)
            #    print("tamanho:", len(ninho))
                #time.sleep(1)
                while (len(ninho) > ((self.tamanhoGene * self.dimensoes) + self.dimensoes + 1)):
                    ninho.pop()
            #print("Ninho após ajuste completo\n:", ninho)
            # time.sleep(1)
            self.populacao.append(ninho)
        #FECHA O FOR
        self.populacao[0] = self.melhorIndividuoDoAG # Salvar o melhor ninho para garantir o elitismo, troca o indivíduo 0
        print("melhorIndividuoDoAG na Avaliar AG:", self.melhorIndividuoDoAG)
        tempoExecucaoAvaliarAG = time.time() - tempoExecucaoAvaliarAG
        print("Tempo da função Avaliação AG:", tempoExecucaoAvaliarAG)



    #Avalia a solução produzida, associando uma aptidão ao ninho
    #e faz o controle dos limites passados - BUSCA CUCO
    def avaliar(self,ninho):
        """
            Avalia a solução produzida, associando uma aptidão ao ninho
            e faz o controle dos limites passados
        """
        ##Ajusta os valores de x e y para ficarem dentro dos limites
        #for dimensao in range(0,self.dimensoes):
        #    if ninho[len(ninho)-1] < self.limiteInferior:
        #        ninho[len(ninho)-1] = self.limiteInferior
        #    elif ninho[len(ninho)-1] > self.limiteSuperior:
        #        ninho[len(ninho)-1] = self.limiteSuperior
        #Ajustar o valor de cada genótipo, referente a cada dimensão para ficar
        #dentro dos limites

        indiceGenotipo = self.tamanhoGene ## Posição onde está o primeiro genótipo -> 21
        indiceInicioGene = 0
        indiceFimGene = self.tamanhoGene - 1
        num_bin = []  # Limpa o vetor para o próximo genótipo
        for dimensao in range(0,self.dimensoes):
            if ninho[indiceGenotipo] < self.limiteInferior:
                #preciso atualizar a linha toda!
                ninho[indiceGenotipo] = self.limiteInferior
                num_bin = bin(self.limiteInferior).replace('0b', '' if self.limiteInferior < 0 else '+').replace('+','').zfill(self.tamanhoGene)  # Converte para binário
                i=0
                for j in range(indiceInicioGene,indiceFimGene+1):
                    ninho[j] = num_bin[i]
                    i+=i
                num_bin = [] # Limpa o vetor para o próximo genótipo
            elif ninho[indiceGenotipo] > self.limiteSuperior:
                #preciso atualizar a linha toda!
                ninho[indiceGenotipo] = self.limiteSuperior
                num_bin = bin(self.limiteSuperior).replace('0b', '' if self.limiteSuperior < 0 else '+').replace('+','').zfill(self.tamanhoGene)  # Converte para binário
                i=0
                for j in range(indiceInicioGene,indiceFimGene+1):
                    ninho[j] = num_bin[i]
                    i+=i
                num_bin = [] # Limpa o vetor para o próximo genótipo
            #Atualizar os indices
            indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
            indiceFimGene = indiceFimGene + self.tamanhoGene + 1
            indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42
        #Ultima posicao guarda a aptidão

        vetorGene = self.retornaVetorGene(ninho)
        aptidao = calculaAptidao(vetorGene, self.dadosBrutos, self.matrizTexto,self.vetorTFIDF)
        ninho[len(ninho)-1] = aptidao
        #ninho.append(aptidao)
        return ninho

    #OK
    def vooLevy(self,beta,Xmelhor,Xi):
        # Para calcular essa distribuição, precisamos de várias variáveis:
        # U; V ; BETA; SigmaU e SigmaV = 1

        ## A principio certo SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        #print("Sigma Antes:",SigmaU)
        #Verificar se gamma é custoso de ser calculado
        #Adicionar o salto de forma mais rara
        SigmaU = np.power(SigmaU, (1 / beta)) * 1000000  # Ajuste
        U = random.gauss(0,SigmaU)
        V = random.gauss(0,1)
        L = (U / (np.power(np.fabs(V), 1 / beta))) * (Xmelhor - Xi)
        #print("SIGMA",SigmaU)
        #print("U",U)
        #print("V",V)
        #print("L(Beta):",L)
        #time.sleep(1)
        return L

    #Função que gera a pertubação causada pelo Cuco, a partir da melhor posição
    def geraCucoPorLevy(self,ninho):
        #Primeiro, temos duas buscas, local e global.
        # A local é:
        # X(t+1) = X(t) + alfa(1) * s[entre 0 e 1] * H(Pa - E) * (Xp - Xq)
        #indiceP = random.randint(1,self.tamanhoPop-1)
        #indiceQ = random.randint(1, self.tamanhoPop - 1)
        # X(t+1) = X(t) + 1*random.uniform(0,1)*np.heaviside(self.Pa - random.uniform(0,1),0.5) * (self.populacao[indiceP][dim] - self.populacao[indiceQ][dim])
        # A global é:
        #Vamos focar na global, já que a outra tem elementos de DE
        # X(t+1) = X(t) + alfa*L(beta), aqui alfa deve ser a escala do salto

        novoNinho = [[] for i in range(0, (self.dimensoes*self.tamanhoGene)+(self.dimensoes)+1 )]# um gene cada dimensão um espaço para cada genotipo mais aptidao
        #Atualiza cada ovo ou soluções para compor a nova

        #Vamos adicionar variação, a busca local deveria ser mais frequente
        #logo:
        if random.randint(1,100) < 50:#Busca global
            #Indices iniciais, é preciso calcular os genes de cada dimensão
            indiceGenotipo = self.tamanhoGene  ## Posição onde está o primeiro genótipo -> 21
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1
            for i in range(0,self.dimensoes):
                #vamos ter que trocar os indices, e calcular o genótipo em binário
                L = self.vooLevy(self.beta,self.populacao[self.tamanhoPop-1].__getitem__(indiceGenotipo),ninho[indiceGenotipo])
                #print("GLOBAL:", L)
                #Se passar dos limites, coloque dentro de novo
                if int(ninho[indiceGenotipo] + self.alfa*L) >= self.limiteSuperior:
                    novoNinho[indiceGenotipo] = self.limiteSuperior
                elif int(ninho[indiceGenotipo] + self.alfa*L) <= self.limiteInferior:
                    novoNinho[indiceGenotipo] = self.limiteInferior
                else:
                    novoNinho[indiceGenotipo] = int(ninho[indiceGenotipo] + self.alfa*L)#Converte para inteiro

                num_bin=bin(novoNinho[indiceGenotipo]).replace('0b', '' if novoNinho[indiceGenotipo] < 0 else '+').replace('+','').zfill(self.tamanhoGene)#Converte para binário
                #print("num_bin:",num_bin)
                #time.sleep(5)
                #print("NUM_BIN:",num_bin)
                #print("Genótipo aqui:",novoNinho[indiceGenotipo])
                k = 0
                for j in range(indiceInicioGene, indiceFimGene + 1):
                    novoNinho[j] = num_bin[k]
                    k = k + 1
                num_bin = []  # Limpa o vetor para o próximo genótipo
                #ponto[i] = ponto[i] + L

                #print("indices:")
                #print("indiceInicioGene", indiceInicioGene)
                #rint("indiceFimGene", indiceFimGene)
                #print("indiceGenotipo", indiceGenotipo)
                #print("NOVO ninho:",novoNinho)

                # Atualizar os indices
                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42

        else:#Busca local

            # Indices iniciais, é preciso calcular os genes de cada dimensão
            indiceGenotipo = self.tamanhoGene  ## Posição onde está o primeiro genótipo -> 21
            indiceInicioGene = 0
            indiceFimGene = self.tamanhoGene - 1

            indiceP = random.randint(1, self.tamanhoPop - 1)
            indiceQ = random.randint(1, self.tamanhoPop - 1)
            while indiceP==indiceQ:     #Pra serem soluções diferentes
                indiceQ = random.randint(1, self.tamanhoPop - 1)
            for i in range(0, self.dimensoes):

                # 1*random.uniform(0,1)*np.heaviside((self.Pa)/100 - random.uniform(0,1),0.5)*(self.populacao[indiceP][i]-self.populacao[indiceQ][i])
                treco = 1 * random.uniform(0, 1) * np.heaviside((50) / 100 - random.uniform(0, 1), 0.5) * (self.populacao[indiceP][indiceGenotipo] - self.populacao[indiceQ][indiceGenotipo])
                #print("LOCAL:", treco)

                if int(ninho[indiceGenotipo] + treco) >= self.limiteSuperior:
                    novoNinho[indiceGenotipo] = self.limiteSuperior
                elif int(ninho[indiceGenotipo] + treco) <= self.limiteInferior:
                    novoNinho[indiceGenotipo] = self.limiteInferior
                else:
                    novoNinho[indiceGenotipo] = int(ninho[indiceGenotipo] + treco)  # Converte para inteiro

                num_bin = bin(novoNinho[indiceGenotipo]).replace('0b','' if novoNinho[indiceGenotipo] < 0 else '+').replace('+','').zfill(self.tamanhoGene)  # Converte para binário
                k = 0
                for j in range(indiceInicioGene, indiceFimGene + 1):
                    novoNinho[j] = num_bin[k]
                    k = k + 1
                num_bin = []  # Limpa o vetor para o próximo genótipo

                # Atualizar os indices
                indiceInicioGene = indiceInicioGene + self.tamanhoGene + 1
                indiceFimGene = indiceFimGene + self.tamanhoGene + 1
                indiceGenotipo = indiceGenotipo + self.tamanhoGene + 1  # Atualiza para ir para a próxima -> 21 -> 42

            #print("LOCAL:", ponto)


        #Faz o ajuste do novo ninho e calcula sua aptidão
        novoNinho = self.avaliar(novoNinho)
        #print("NOVO ninho:", novoNinho)
        #time.sleep(2)
        return novoNinho

    #Função que coloca valores aleatórios para os piores ninhos -> os primeiros
    def abandonaPa(self,indicePa):
        #complementoPb = 100 - self.Pa  # Pb
        #indicePa = (int)((self.tamanhoPop * complementoPb) / 100) - 1  # Indice onde começam as piores soluções
        tempoExecucao = time.time()
        for k in range(0, indicePa):
            #print("Olhar o K:",k)
            #print("Ninho Antigo",self.populacao[k])
            self.populacao[k] = self.geraNinhoAleatorio(self.populacao[k])
            #print("Ninho Novo", self.populacao[k])
            #time.sleep(1)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo da função abandonaPa:",tempoExecucao)

    #OK
    def geraNinhoAleatorio(self, ninhoAleatorio):
        #tempoExecucao = time.time()
        #ninhoAleatorio = [[] for i in range(self.dimensoes+1)]
        #print("Ninho enviado", ninhoAleatorio)
        ninhoAleatorio.clear() # preciso limpar ele para que ele receba o novo ninho
        for dim in range(0,self.dimensoes):
            XN = random.randint(self.limiteInferior, self.limiteSuperior-random.randint(0,1000))  # gera um número aleatório
            #print("Número aleatório gerado:",XN)
            num_bin = bin(XN).replace('0b', '' if XN < 0 else '+').replace('+','').zfill(self.tamanhoGene) # Converte para binário
            #print("num_bin",num_bin)
            for bit in num_bin:
                ninhoAleatorio.append(bit)
            ninhoAleatorio.append(XN)
        #print("Tamanho do gene",self.tamanhoGene)
        vetorGene = self.retornaVetorGene(ninhoAleatorio)
        # print("Vetor gene:",vetorGene)

        aptidao = calculaAptidao(vetorGene, self.dadosBrutos, self.matrizTexto,self.vetorTFIDF)
        ninhoAleatorio.append(aptidao)

        #print("GERA ninho aleatorio",ninhoAleatorio)

        #Não faz sentido
        #ninhoAleatorio = self.avaliar(ninhoAleatorio)

        #print("GERA ninho aleatorio", ninhoAleatorio)
        #time.sleep(5)


        #print("GERA ninho aleatorio",ninhoAleatorio)
        #time.sleep(100)
        #tempoExecucao = time.time() - tempoExecucao
        #print("Tempo da função gera Ninho Aleatorio:", tempoExecucao)
        return ninhoAleatorio


def funcaoAptidao(alfa, beta, acuracia, numeroAtributos):
    # alfa = 1 -> controlam a função de aptidão
    # beta = 1
    # Trocar a função manualmente aqui
    resultado = funcoes.funcaoAptidaoBCG(alfa, beta, acuracia, numeroAtributos)
    return resultado


# Calcula o valor do fenotipo ou APTIDAO a partir do genotipo
def calculaAptidao(vetorGene, dadosBrutos, matrizTexto, vetorTFIDF):
    estadoInicial = 5

    #tempoExecucaoCalcula = time.time()
    # Preciso pegar os valores de genótipo, presente nas posições
    # tamanhoGene*i; i=1


    # acuracia = ninho[len(ninho)-1]

    # Para calcular a aptidão, precisamos primeiro da acurácia,
    # logo, precisamos da matriz modificada a partir do vetorGene(0s e 1s)
    # e vamos precisar gerar o modelo com os conjuntos a serem preditos
    #tempoExecucaoCalcula = time.time()
    matrizModificada = manipulacaoTexto.transformaMatriz(matrizTexto, vetorTFIDF, vetorGene)
    #tempoExecucaoCalcula = time.time() - tempoExecucaoCalcula
    #print("Tempo da função transformaMatriz:", tempoExecucaoCalcula)
    #tempoExecucaoCalcula = time.time()
    x_treino, x_teste, y_treino, y_teste = manipulacaoTexto.constroiModelo(dadosBrutos, matrizModificada,estadoInicial)
    #tempoExecucaoCalcula = time.time() - tempoExecucaoCalcula
    #print("Tempo da função constroiModelo:", tempoExecucaoCalcula)
    #tempoExecucaoCalcula = time.time()
    acuracia = manipulacaoTexto.classificadorME(x_treino, x_teste, y_treino, y_teste)
    #tempoExecucaoCalcula = time.time() - tempoExecucaoCalcula
    #print("Tempo da funçãoclassificadorME:", tempoExecucaoCalcula)
    # O número de atributos válidos é calculado com os "1" presentes no vetorGene
    numeroAtributosSelecionados = 0
    for i in range(0, len(vetorGene)):
        if vetorGene[i] == 1:
            numeroAtributosSelecionados = numeroAtributosSelecionados + 1

    #A aptidão é calculada maximizando a acurácia e e minimizando o numero de atributos
    #Isso está feito na funcaoAptidao

    aptidao = funcaoAptidao(1, 1, acuracia, numeroAtributosSelecionados)
    #tempoExecucaoCalcula = time.time() - tempoExecucaoCalcula
    #print("Tempo da função calcula Aptidão:", tempoExecucaoCalcula)
    #time.sleep(5)
    return aptidao


# Calcula o valor do genotipo a partir do vetorGene
# OK
def calculaGenotipo(tamanhoGene, individuo, inicioGene, fimGene):
    soma = 0
    # Tira -1 porque um é da conversão
    base = 2 ** (tamanhoGene - 1)  # se tamanhoGene = 5, base = 8; tamanhoGene = 8, base = 2^6=64
    # print("Dados da calcula genotipo:")
    # print("BASE:",base)
    # time.sleep(100)
    # print("Tamanho Gene:",tamanhoGene)
    # print("Ninho:",individuo)
    # print("Indices:",inicioGene,fimGene)
    # time.sleep(4)
    for j in range(inicioGene, fimGene + 1):
        # print("J:", j)
        if individuo[j] == '1':
            soma = soma + base
        base = base / 2
    soma = int(soma)
    # print("SOMAAAA:",soma)
    # time.sleep(3)
    return soma

#Ok
def encontraMedia(self):
    media = 0
    for individuo in self.populacao:
        media = media + individuo[len(individuo)-1]
        #print("Media",media)
    media = media/len(self.populacao)
    #print("media",media)
    #print("tamanho",len(self.populacao))
    return media

#OK
def exibePopulacao(self):
    i = 0
    for ninho in self.populacao:
        print("Indivíduo[", i, "]:")
        print(ninho, "\n\t -> Aptidão:", ninho[len(ninho)-1], "\n")
        i += 1

#OK
def exibeMelhorIndividuo(self):
    ninhoBom = self.populacao[self.tamanhoPop-1]
    print("Melhor aptidão encontrada:", ninhoBom[len(ninhoBom)-1])
    return self.populacao[self.tamanhoPop-1]


def main():
    novoNinho_file = open("saidas/novo_ninho_agSentimento.txt", "w")  # Gravação
    melhorNinho_file = open("saidas/melhor_agSentimento.txt", "w")  # Gravação
    medias_file = open("saidas/media_agSentimento.txt", "w")  # Gravação

    # Colocar aqui a função para fazer a limpeza e para salvar os dados pós transformação TF-IDF


    #Função para carregar os dados, a matriz texto (amostras X atributos) e
    #o vetor que possui os atributos
    tempoExecucao = time.time()
    dadosBrutos,matrizTexto,vetorTFIDF = manipulacaoTexto.carregaDados()
    numeroDeAtributosTotal = len(vetorTFIDF.vocabulary_)  ## Quantidade de atributos/colunas
    print("MATRIZ (linhas:colunas): \n", matrizTexto.shape, "\n")
    print("Número de atributos inicial (total):",numeroDeAtributosTotal,"\n Resto:",int(numeroDeAtributosTotal%10))
    numpy.set_printoptions(threshold=sys.maxsize)  ##Exibe a matriz por inteiro
    print(matrizTexto)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução carregar dados:", tempoExecucao)

    ativaAG = 1 #Controla se o AG vai ser executado também

    limiteInferior = 0
    limiteSuperior = 1023 #( 500 TamanhoGene fica em 9 ->010101001) (1023 o tamanhoGene fica em 10)
    dimensoes = int(numeroDeAtributosTotal/10) #Depende do número de atributos total
    restoAtributos = int(numeroDeAtributosTotal%10) #Deixar os últimos atributos sempre selecionados!
    tamanhoPop = 100
    Pa = 10
    alfa = 1
    beta = 2
    pc = 70
    pm = 10
    maxGeracoes = 1000
    geracoesEstagnadas = 50

    tempoExecucao = time.time()
    busca_Cuco_Genetico = buscaCucoGenetico(limiteInferior,limiteSuperior,dimensoes,restoAtributos,
                                            tamanhoPop,Pa,alfa,beta,pc,pm,maxGeracoes,geracoesEstagnadas,
                                            dadosBrutos,matrizTexto,vetorTFIDF)
    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução iniciação BCG:", tempoExecucao)

    #Todos os ninhos são avaliados inicialmente
    #exibePopulacao(self=busca_Cuco_Genetico)

    estagnacao = 0
    PB = int((busca_Cuco_Genetico.tamanhoPop * (100 - busca_Cuco_Genetico.Pa)) / 100)  # 1 - Pa -> 75
    #complementoPb = 100 - busca_Cuco_Genetico.Pa  # Pb
    #indicePa = (int)((busca_Cuco_Genetico.tamanhoPop * complementoPb) / 100) - 1  # Indice onde terminam as piores soluções

    tempoExecucaoTotal = time.time()
    for epoca in range(busca_Cuco_Genetico.maxGeracoes):
        tempoExecucao = time.time()
        ultimoMelhorIndividuo = busca_Cuco_Genetico.populacao[tamanhoPop-1]  # As soluções ficam em ordem crescente, a melhor fica na ultima posição
        ultimaMelhorAptidao = ultimoMelhorIndividuo[len(ultimoMelhorIndividuo)-1]

        #print("ultimo melhor individuo:",ultimoMelhorIndividuo)

        #time.sleep(1)
        #exibePopulacao(self=busca_Cuco)

        #Gerar cuco por voo de Levy -> usar um dos melhores indivíduos como semente
        indiceBom = random.randint(PB,tamanhoPop-1) # Meu I

        #print("Escolhi um bom individuo i=",indiceBom)
        #print("\n->",busca_Cuco.populacao[indiceBom])
        novoNinho = busca_Cuco_Genetico.geraCucoPorLevy(busca_Cuco_Genetico.populacao[indiceBom])
        #time.sleep(1)

        #Salvar o novo ninho descoberto a cada geração
        novoNinho_file.write("%s \n" % (novoNinho) )

        #Escolher o ninho J aleatório
        #indiceJ = random.randint(0, busca_Cuco_Genetico.tamanhoPop - 1)  -> CERTO
        #indiceJ = random.randint(PB, busca_Cuco_Genetico.tamanhoPop - 1)## Teste para acelerar
        #ninhoJ = busca_Cuco_Genetico.populacao[indiceJ]
        #print("J:",indiceJ,"\nNINHO J",ninhoJ)
        #print("NOVO CUCO:", novoNinho)
        #time.sleep(1)


        #Se novoCuco.apt > ninhoJ.apt -> troque, se trocou, ordene! é O(1)
        #print("Novo:",novoNinho[len(novoNinho)-1])
        #print("Ninho J:",ninhoJ[len(ninhoJ)-1])
        #if novoNinho[len(novoNinho)-1] > ninhoJ[len(ninhoJ)-1]:
            #Associa o novo ninho ao ninho J
        #    for i in range(0,busca_Cuco_Genetico.dimensoes+busca_Cuco_Genetico.restoAtributos+1):
        #        busca_Cuco_Genetico.populacao[indiceJ][i] = novoNinho[i]

        #    busca_Cuco_Genetico.populacao = sorted(busca_Cuco_Genetico.populacao,key=lambda ninho: ninho[len(ninho)-1])  # Ordenar de acordo com a qualidade, ordem crescente
        #    novoNinho.clear()#limpar para a proxima epoca
            #print("\n#######################################\n")

        #Parte do algoritmo genético: evolução
        #Executa a cada x gerações - as operações são pesadas
        if (ativaAG==1):
            #Seleção
            busca_Cuco_Genetico.selecao()
            #Cruzamento
            busca_Cuco_Genetico.cruzamento()
            #Mutação
            busca_Cuco_Genetico.mutacao()
            #for indiv in busca_Cuco_Genetico.populacao:
            busca_Cuco_Genetico.avaliarAG()
            #O elistismo seria aplicado aqui, porém a BC já o faz naturalmente
            busca_Cuco_Genetico.populacao = sorted(busca_Cuco_Genetico.populacao, key=lambda ninho: ninho[len(ninho) - 1]) # Ordenar de acordo com a qualidade

        mediaGeracao = encontraMedia(self=busca_Cuco_Genetico)#Pega a média de aptidão da população

        medias_file.write("Geracao: %d \t %f \t %s \n" % (epoca, mediaGeracao, ultimoMelhorIndividuo[len(ultimoMelhorIndividuo)-1]))
        melhorNinho_file.write( "%s\n" % (ultimoMelhorIndividuo) )

        testeAqui = busca_Cuco_Genetico.populacao[busca_Cuco_Genetico.tamanhoPop-1]
        aqui = testeAqui[len(testeAqui)-1] #aqui é a melhor aptidão
        ##Verificação de estagnação
        #if ultimaMelhorAptidao == busca_Cuco_Genetico.populacao[0]:
        if (ultimaMelhorAptidao == aqui) or (aqui>=0.96): ##Se não vai ficar com um ajuste muito fino
            estagnacao = estagnacao + 1
        else:
            estagnacao = 0
        if estagnacao == busca_Cuco_Genetico.geracoesEstagnadas:
            print("\n\n## -- ESTAGNOU! -- ##")
            break


        #exibePopulacao(self=busca_Cuco_Genetico)
        print("##ÉPOCA:", epoca)
        print("-- ultima melhor aptidão:", ultimaMelhorAptidao)
        tempoExecucao = time.time() - tempoExecucao
        print("Tempo de execução da época:", tempoExecucao)
        print("--------------------------------")
        #time.sleep(1)

    tempoExecucaoTotal = time.time() - tempoExecucaoTotal
    exibePopulacao(self=busca_Cuco_Genetico)
    umNinho = buscaCucoGenetico.retornaVetorGene(busca_Cuco_Genetico,busca_Cuco_Genetico.populacao[busca_Cuco_Genetico.tamanhoPop-1])
    print("Ninho modificado:",umNinho)
    somaUm=0
    for bit in umNinho:
        if bit==1:
            somaUm = somaUm+1
    porcentagemAtributo=(10*somaUm)/(busca_Cuco_Genetico.dimensoes + busca_Cuco_Genetico.restoAtributos)
    print("Atributos selecionados (%):",porcentagemAtributo)
    print("Tempo de execução TOTAL:",tempoExecucaoTotal," Minutos:",tempoExecucaoTotal/60)
    print("Número de épocas:", epoca)
    print("Média da última geração:", mediaGeracao)
    print("Melhor ninho encontrado:", exibeMelhorIndividuo(self=busca_Cuco_Genetico))

    novoNinho_file.close()
    melhorNinho_file.close()
    medias_file.close()
    return 0

if __name__ == '__main__':
    main()