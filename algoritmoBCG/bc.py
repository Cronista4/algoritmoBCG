import time
import random
import funcoes
import numpy as np
import math

ponto = [0,0]

def funcaoAptidao(ninho,dimensoes):
    #Trocar a função manualmente aqui
    resultado = funcoes.rastriginCompleta(ninho,dimensoes)
    return resultado

# Calcula o valor do fenotipo ou APTIDAO a partir do genotipo
def calculaAptidao(ninho,dimensoes):
    aptidao = funcaoAptidao(ninho,dimensoes)
    return aptidao

class buscaCuco():

    def __init__(self,limiteInferior,limiteSuperior,dimensoes,tamanhoPop,Pa,alfa,beta,maxGeracoes,geracoesEstagnadas):
        self.limiteInferior = limiteInferior
        self.limiteSuperior = limiteSuperior
        self.dimensoes = dimensoes
        self.tamanhoPop = tamanhoPop # Deve ficar entre 20 e 100
        self.Pa = Pa
        self.alfa = alfa # Geralmente 1
        self.beta = beta # Geralmente 0<=b<=2
        self.maxGeracoes = maxGeracoes # De mil até 100k
        self.geracoesEstagnadas = geracoesEstagnadas # Talvez 100

        self._iniciaPopulacao()

    def _iniciaPopulacao(self):
        # Inicia a população de ninhos
        self.populacao = [[] for i in range(self.tamanhoPop)]

        #Vamos considerar que cada ninho tenha exatamente dois ovos -> x e y que compoem f(x,y) = Z
        for ninho in self.populacao:
            #Representa cada ovo
            for dim in range(0,self.dimensoes):
                XN = random.uniform(self.limiteInferior, self.limiteSuperior)  # float
                ninho.append(XN)
            aptidao = calculaAptidao(ninho,self.dimensoes)
            ninho.append(aptidao)

        #ORDENAÇÃO
        self.populacao = sorted(self.populacao,key=lambda ninho: ninho[2])#Ordenar de acordo com a qualidade, ordem crescente

    def avaliar(self,ninho):
        """
            Avalia a solução produzida, associando uma aptidão ao ninho
            e faz o controle dos limites passados
        """
        ##Ajusta os valores de x e y para ficarem dentro dos limites
        for dimensao in range(0,self.dimensoes):
            if ninho[dimensao] < self.limiteInferior:
                ninho[dimensao] = self.limiteInferior
            elif ninho[dimensao] > self.limiteSuperior:
                ninho[dimensao] = self.limiteSuperior
        #Ultima posicao guarda a aptidão
        ninho[self.dimensoes] = calculaAptidao(ninho,self.dimensoes)
        return ninho

    def vooLevy(self,beta,Xmelhor,Xi):
        # Para calcular essa distribuição, precisamos de várias variáveis:
        # U; V ; BETA; SigmaU e SigmaV = 1

        ## A principio certo SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        SigmaU = ( math.gamma(1+beta) * np.sin(beta*np.pi/2)) / (math.gamma((1+beta)/2) * beta *np.power(2,(1-beta)/2) )
        #print("Sigma Antes:",SigmaU)
        #Verificar se gamma é custoso de ser calculado
        #Adicionar o salto de forma mais rara
        if random.randint(0,100) < 5:
            SigmaU = np.power(SigmaU, (1 / beta)) * 1000000  # Ajuste
        else:
            SigmaU = np.power(SigmaU, (1 / beta)) * 1  # Ajuste************
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

        novoNinho = [[] for i in range(0, self.dimensoes+1)]
        #Atualiza cada ovo ou soluções para compor a nova

        #Vamos adicionar variação, a busca local deveria ser mais frequente
        #logo:
        if random.randint(1,100) < 50:#Busca global

            for i in range(0,self.dimensoes):
                L = self.vooLevy(self.beta,self.populacao[0].__getitem__(i),ninho[i])
                novoNinho[i] = ninho[i] + self.alfa*L
                #ponto[i] = ponto[i] + L
                #print("GLOBAL:", L)
            #print("GLOBAL:",ponto)
        else:#Busca local
            indiceP = random.randint(1, self.tamanhoPop - 1)
            indiceQ = random.randint(1, self.tamanhoPop - 1)
            while indiceP==indiceQ:     #Pra serem soluções diferentes
                indiceQ = random.randint(1, self.tamanhoPop - 1)
            for i in range(0, self.dimensoes):
                #1*random.uniform(0,1)*np.heaviside((self.Pa)/100 - random.uniform(0,1),0.5)*(self.populacao[indiceP][i]-self.populacao[indiceQ][i])
                treco = 1*random.uniform(0,1)*np.heaviside((50)/100 - random.uniform(0,1),0.5)*(self.populacao[indiceP][i]-self.populacao[indiceQ][i])
                novoNinho[i] = ninho[i] + treco
                #ponto[i] = ponto[i] + treco
                #print("LOCAL:", treco)
            #print("LOCAL:", ponto)
        #Faz o ajuste do novo ninho e calcula sua aptidão
        novoNinho = self.avaliar(novoNinho)
        #print("NOVO ninho que deve ser diferente:", novoNinho)

        #time.sleep(100)
        return novoNinho

    def abandonaPa(self):
        complementoPb = 100 - self.Pa  # Pb
        indicePa = (int)((self.tamanhoPop * complementoPb) / 100) - 1  # Indice onde começam as piores soluções

        for k in range(indicePa, self.tamanhoPop):
            #print("Olhar o K:",k)
            #print("Ninho Antigo",self.populacao[k])
            self.populacao[k] = self.geraNinhoAleatorio(self.populacao[k])
            #print("Ninho Novo", self.populacao[k])
            #time.sleep(1)
            #print("NOVOS NINHOS RANDOM:", self.populacao[k])


    def geraNinhoAleatorio(self, ninhoAleatorio):
        #ninhoAleatorio = [[] for i in range(self.dimensoes+1)]
        for dim in range(0, self.dimensoes):
            XN = random.uniform(self.limiteInferior, self.limiteSuperior)  # float
            ninhoAleatorio[dim] = XN
        aptidao = calculaAptidao(ninhoAleatorio, self.dimensoes)
        ninhoAleatorio[self.dimensoes] = aptidao
        ninhoAleatorio = self.avaliar(ninhoAleatorio)
        #print("GERA ninho aleatorio",ninhoAleatorio)
        #time.sleep(1)
        return ninhoAleatorio


def encontraMedia(self):
    media = 0
    for individuo in self.populacao:
        media = media + individuo[self.dimensoes]
        #print("Media",media)
    media = media/len(self.populacao)
    #print("media",media)
    #print("tamanho",len(self.populacao))
    return media

def exibePopulacao(self):
    i = 0
    for ninho in self.populacao:
        print("Indivíduo[", i, "]:")
        print(ninho, "\n\t -> Aptidão:", ninho[self.dimensoes], "\n")
        i += 1

def exibeMelhorIndividuo(self):
    return self.populacao[0]

def main():
    text_file = open("saidas/saida_bc.txt", "w")  # Gravação
    medias_file = open("saidas/media_bc.txt", "w")  # Gravação
    # limiteInferior, limiteSuperior, dimensoes, tamanhoPop, Pa,alfa, beta, maxGeracoes, geracoesEstagnadas

    limiteInferior = -1000
    limiteSuperior = 1000  # ( 500 TamanhoGene fica em 10 ->+010101001)
    dimensoes = 10
    tamanhoPop = 100
    Pa = 10
    alfa = 1
    beta = 2
    maxGeracoes = 10000
    geracoesEstagnadas = 500
    busca_Cuco = buscaCuco(limiteInferior,limiteSuperior,dimensoes,
                           tamanhoPop,Pa,alfa,beta,maxGeracoes,geracoesEstagnadas)#TODOS são avaliados inicialmente
    exibePopulacao(self=busca_Cuco)

    estagnacao = 0
    tempoExecucao = time.time()
    for epoca in range(busca_Cuco.maxGeracoes):
        ultimoMelhorIndividuo = busca_Cuco.populacao[0]  # As soluções ficam em ordem crescente, a melhor fica na posição 0
        #exibePopulacao(self=busca_Cuco)

        #Gerar cuco por voo de Levy -> usar um dos melhores/pior indivíduo como semente
        indiceBom = random.randint(1,75) # Meu I
        #print("Escolhi um bom individuo i=",indiceBom)
        #print("\n->",busca_Cuco.populacao[indiceBom])
        novoNinho = busca_Cuco.geraCucoPorLevy(busca_Cuco.populacao[indiceBom])
        #Salvar o novo ninho descoberto a cada geração
        text_file.write("%f\t %f \t %f \n" % (novoNinho[0], novoNinho[1], novoNinho[2]))

        #Escolher o ninho J aleatório
        indiceJ = random.randint(0, busca_Cuco.tamanhoPop - 1)
        ninhoJ = busca_Cuco.populacao[indiceJ]
        #print("J:",indiceJ,"\nNINHO J",ninhoJ)
        #Se novoCuco.apt < ninhoJ.apt -> troque, se trocou, ordene! é O(1)
        if novoNinho[busca_Cuco.dimensoes] < ninhoJ[busca_Cuco.dimensoes]:
            #print("NOVO CUCO:", novoNinho)
            for i in range(0,busca_Cuco.dimensoes+1):
                busca_Cuco.populacao[indiceJ][i] = novoNinho[i]

            busca_Cuco.populacao = sorted(busca_Cuco.populacao,key=lambda ninho: ninho[2])  # Ordenar de acordo com a qualidade, ordem crescente
            novoNinho.clear()#limpar para a proxima epoca
            #print("\n#######################################\n")

        # Abandone a porção Pa -> Gere novas soluções aleatórias aqui
        busca_Cuco.abandonaPa()
        # Ordene de novo de acordo com a aptidão e salve o melhor
        busca_Cuco.populacao = sorted(busca_Cuco.populacao,key=lambda ninho: ninho[2])  # Ordenar de acordo com a qualidade, ordem crescente
        #exibePopulacao(self=busca_Cuco)

        mediaGeracao = encontraMedia(self=busca_Cuco)#Pega a média de aptidão da população

        medias_file.write("Geração: %d \t %f \t %s \n" % (epoca, mediaGeracao, ultimoMelhorIndividuo[busca_Cuco.dimensoes]))

        ##Verificação de estagnação
        if ultimoMelhorIndividuo == busca_Cuco.populacao[0]:
            estagnacao = estagnacao + 1
        else:
            estagnacao = 0
        if estagnacao == busca_Cuco.geracoesEstagnadas:
            print("\n\nESTAGNOU!!!!!!!!!")
            break

    tempoExecucao = time.time() - tempoExecucao
    print("Tempo de execução:",tempoExecucao)
    print("Número de épocas:", epoca)
    print("Média da última geração:", mediaGeracao)
    print("Melhor indivíduo encontrado:", exibeMelhorIndividuo(self=busca_Cuco))
    text_file.close()
    medias_file.close()
    return 0

if __name__ == '__main__':
    main()