#!/usr/bin/env python
# coding: utf-8

# In[1]:


#simplex duas fases da Ju Salessi e Pedro
def simplex(A, b, c, index_vectorA): #funcao que realiza o metodo simplex,  
    #A=matriz dos coeficientes das restricoes.
    #b=vetor de recursos
    #c=vetor de custos
    # index_vectorA=vetor dos indices da matriz A na iteracao atual
    flag = False
    j = 0
    while flag == False:
        j+=1;

        m = A.shape[0] #numero de linhas de A
        n = A.shape[1] #numero de colunas de A

        Asplit = numpy.hsplit(A, [n-m]) #separacao da matriz A em 2
        B = Asplit[1] #B=parte basica da matriz A
        N = Asplit[0] #N= parte nao basica da matriz A

        xb = numpy.linalg.solve(B, b) #calculo da solucao basica pelo sistema B*xb=b
        
        #calculo dos custos relativos

        cb = numpy.split(c, [c.shape[0]-m]) #separacao de cb (custos de b) do vetor de custos c 

        Bt = B.transpose() #transpondo a matriz B, Bt= B transposta
        
        #vetor multiplicador simplex

        lamb = numpy.linalg.solve(Bt, cb[1]) #lamb= vetor mult simplex, calculado pelo sistema Bt*lamb=cb
        
        #custos relativos 
        
        cr = numpy.array([]) #criacao do vetor custo relativo

        for i in range(0, n-m):
            aux = c[i] - numpy.matmul(lamb.transpose(), N.transpose()[i]) #calculo de cada valor de cr pela expressao ci-lambT*ai////ai=coluna i de N
            cr = numpy.append(cr, aux) # preenche vetor cr com cada valor de cri

    
        min_index = numpy.argmin(cr) #indice do menor valor de cr

        if ((cr >= 0).all()): #se todos valores de custo relativo sao maiores que zero a solucao e otima 
            final = numpy.matmul(cb[1].transpose(),xb) # valor da sol otima = matriz custos de b *  xb
            
            V = numpy.zeros(n) #preencho um vetor que vai representar x de tamanho n de zeros pois todos xn tem valor zero
            for i in range (0, m):
                V[index_vectorA[i+n-m]] = xb[i] #se o indice do vetor for algum de xb eu troco pelo valor de xb
            flag = True            
            return (final, V, A, c, index_vectorA) #a funcao retorna os valores da solucao final, o vetor com os ultimos xbexn na ordem correta, o vetor custos na ordem atual, e os indices pra saber qual eh a ordem atual
            break
        else:
            #se a solucao nao eh otima calculo da direcao simplex

            y = numpy.linalg.solve(B, A[:,min_index])  #y=direcao simplex, B*y= coluna de N relativa ao minimo de cr 
            if ((y <= 0).all()): #se todos termos de y forem negativos a solucao final eh ilimitada
                #solução ilimitada
                flag = True
                return (None, None, None, None, None)
                break
            else: #determinacao do passo e variavel a sair da base 
                epsilon_arr = numpy.array([]) #criacao do vetor de epsilon
    
                epsilon_index = 0; #indice do menor epsilon(o que vai sair da base
                minimum = sys.maxint #valor min= cte alta

                for i in range(xb.shape[0]): #preenchimento do vetor de epsilon
                    if (y[i] > 0):#os valores possiveis de epsilon so sao calculados se a direcao simplex eh maior que zero
                        aux = xb[i]/y[i]  #calculo dos possiveis valores de epsilon 
                        epsilon_arr = numpy.append(epsilon_arr, aux) #coloca valores de epsiolon no vetor epsiolon
                        if (aux < minimum):  #salva o indice do epsilon oficial, que eh o minimo do vetor de epsilons
                            epsilon = aux
                            minimum = aux
                            epsilon_index = i
             

          
            aa = min_index #indice do que entra na base
            bb = epsilon_index+N.shape[1] #indice do que sai da base
            
            A[:,[aa,bb]] = A[:,[bb,aa]] #inverte as colunas que entra e sai da base
            index_vectorA[[aa,bb]] = index_vectorA[[bb,aa]] #inverte tbm no vetor de indices
           
            Asplit = numpy.hsplit(A, [N.shape[1]]) #divide A de novo
            B = Asplit[0] #novo B
            N = Asplit[1] #novo N
            
            c[[aa,bb]] = c[[bb,aa]] #novos custos que eh o vetor antigo com a ordem trocada            


# In[4]:


#inicio do nosso programa maroto e implementacao das 2 fases 

import StringIO, numpy, sys #bibliotecas usadas

data = open('A.txt').read()
A = numpy.genfromtxt(StringIO.StringIO(data), delimiter = ' ') #pega A de um arquivo

data = open('b.txt').read()
b = numpy.genfromtxt(StringIO.StringIO(data), delimiter = ' ') #pega b de um arquivo

data = open('c.txt').read()
c = numpy.genfromtxt(StringIO.StringIO(data), delimiter = ' ') #pega c de um arquivo

data = open('mn.txt').read() 
m,n = numpy.genfromtxt(StringIO.StringIO(data), delimiter = ' ') #pega m e n de um arquivo
m = int(m)
n = int(n)

for i in range(b.shape[0]): #se algum termo de b for menor que zero a linha inteira correspondente eh multiplicda por menos 1 
    if (b[i] < 0):
        b[i] = -b[i]
        for j in range(n):
            A[i][j] = -A[i][j]

n_zeros = 0 #numero de termos que tem custo relativo igual a zero(ou seja variavel de folga
for i in c:
    if (i == 0): #se for igual a zero soma mais um no numero de zeros 
        n_zeros += 1
        
index_vectorA = numpy.array(range(0,A.shape[1])) #indice de A no comeco do programa eh 0 1 2 3....ordem normal
        
if (n_zeros != m): # se tem menos variaveis de folga que restricoes (ou seja tem alguma restricao de igualdade
    #coloca variaveis artificiais
    B = numpy.concatenate((A,numpy.identity(int(m))),axis=1) #concatena uma matriz identidade em B que sao as variaveis de folga
    c1 = numpy.concatenate((numpy.zeros(int(n)),numpy.ones(m))) #acrescenta m variaveis de custo zero em c
    index_vectoraux = numpy.concatenate((index_vectorA,range(n,m+n))) #aumenta o vetor de indices
    
    final, V, B, c1, index_vectoraux = simplex(B,b,c1,index_vectoraux) #roda simplex da fase 1
    if (final == 0 ): #se o simplex nao for indeterminado e tiver solucao otima=0
        c2 = c.copy() #cria um vetor de custos auxiliar
        j = 0; 
        
        for i in range(n+m):    #tira colunas extras(acrescentadas na fase 1) em A e B e c
            if(index_vectoraux[i] < n): 
                A[:,j] = B[:,i]
                c[j] = c2[index_vectoraux[i]]
                index_vectorA[j] = index_vectoraux[i]
                j+=1;         
        final, V, A,c, index_vectorA = simplex(A,b,c,index_vectorA) #roda simplex na fase 2
        if (final != None):
            print("solucao otima:")
            print(final)
            print("valores das variaveis")
            print(V)
        else:
            print ("solucao ilimitada")
       
        
    else:
        print("infactivel")
    
else: 
    if((A[:,-(n_zeros):] < 0).any()): #se alguma das variaveis de folga estiver com valor negativo em A
        #coloca variaveis artificiais
        B = numpy.concatenate((A,numpy.identity(int(m))),axis=1) #concatena uma matriz identidade em B que sao as variaveis de folga
        c1 = numpy.concatenate((numpy.zeros(int(n)),numpy.ones(m))) #acrescenta m variaveis de custo zero em c
        index_vectoraux = numpy.concatenate((index_vectorA,range(n,m+n))) #aumenta o vetor de indices
       
        final, V, B, c1, index_vectoraux = simplex(B,b,c1,index_vectoraux) #roda simplex da fase 1
        if (final == 0):#se o simplex nao for indeterminado e tiver solucao otima=0
            c2 = c.copy() #cria um vetor de custos auxiliar
            
            j = 0;
            
            for i in range(n+m):     #tira colunas extras(acrescentadas na fase 1) em A e B e c           
                if(index_vectoraux[i] < n):
                    A[:,j] = B[:,i]
                    c[j] = c2[index_vectoraux[i]]
                    index_vectorA[j] = index_vectoraux[i]
                    j+=1;         
            final, V, A,c, index_vectorA = simplex(A,b,c,index_vectorA) #roda simplex na fase 2
            if (final != None):
                print("solucao otima:")
                print(final)
                print("valores das variaveis")
                print(V)
            else:
                print ("solucao ilimitada")
        else:
            print("infactivel")
    
    else:
     #roda simplex normalzao
        final, V, A, index_vectorA = simplex(A,b,c,index_vectorA)
        if (final != None):
            print("solucao otima:")
            print(final)
            print("valores das variaveis")
            print(V)
        else:
            print ("solucao ilimitada")


# In[106]: