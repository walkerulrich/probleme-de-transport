import numpy as np
import copy as copy
def Split1(chaine):
    Res=[]
    s=chaine.split(" ")
    for i in s:
        if i!="":
            Res.append(i)
    return Res

def lectureRO(a):
    f=open(a,"r")
    s=f.readlines()
    f.close()
    res=[]
    res2=[]
    for i in range(len(s)):
        if s[i][len(s[i])-1]=="\n" :
            a = s[i][0:len(s[i]) - 1:1]
        else:
            a=s[i][0:len(s[i]):1]
        b=Split1(a)
        res2.append(b)

    #On élimine les lignes vides
    for i in res2:
        if i != []:
            res.append(i)

    #Conversion des chaines de caractères en entier
    for i in range(len(res)):
        for j in range(len(res[i])):
            res[i][j]=int(res[i][j])
    nbrPC=[]
    Commandes=[]
    Provisions=[]
    Matrice_de_cout=[]
    for i in range(len(res)):
        if i == 0:
            nbrPC=res[i]

        elif i == len(res)-1:
            Commandes=res[i]
        else:
            Matrice_de_cout.append(res[i][0:len(res[i])-1:])
            Provisions.append(res[i][len(res[i])-1])

    return  Commandes,Provisions,np.array(Matrice_de_cout)

def afficher_matrice(matrice):
    # Affichage des indices de colonnes
    print("\t", end="")
    ind_col=""
    for j in range(len(matrice[0])):
        ind_col+= "C"+str(j+1)+"\t"
    print(ind_col,end="")
    print()

    # Affichage de la matrice
    for i, ligne in enumerate(matrice):
        ligne_matrice = ""
        ligne_matrice += "P"+str(i+1)+"\t"
        for val in ligne:
            ligne_matrice += str(val)+"\t"
        print(ligne_matrice)




def Nord_Ouest(Provisions,Commandes):
    nb_l = len(Provisions)
    nb_c = len(Commandes)
    dif=0
    i, j = 0, 0
    matrice_res = [[0] * nb_c for _ in range(nb_l)]
    while i < nb_c  and j < nb_l:
        dif= min(Commandes[i],Provisions[j])
        matrice_res[j][i] = dif
        Commandes[i] -= dif
        Provisions[j] -= dif
        if Commandes[i] == 0:
            i += 1
        else:
            j += 1
    return np.array(matrice_res)

def Balas_Hammer(matrice_couts, c, p, a=False):
    n = len(p)
    m = len(c)
    matrice_res = np.zeros((n, m), dtype=int)
    provisions = p.copy()
    commandes = c.copy()
    couts = matrice_couts.copy()
    iterations = 0

    while sum(commandes) > 0 and sum(provisions) > 0:
        iterations += 1
        penalites_lignes = np.zeros(couts.shape[0], dtype=int)
        penalites_colonnes = np.zeros(couts.shape[1], dtype=int)

        if a:
            print("******")
            print("ITERATION", iterations)

        sorted_couts_lignes = np.sort(couts, axis=1)
        sorted_couts_colonnes = np.sort(couts, axis=0)

        for i in range(couts.shape[0]):
            if  provisions[i] > 0:
                penalites_lignes[i] = sorted_couts_lignes[i][1] - sorted_couts_lignes[i][0] if sorted_couts_lignes[i, :].shape[0] > 1 else sorted_couts_lignes[i][0]

        for j in range(couts.shape[1]):
            if commandes[j] > 0:
                penalites_colonnes[j] = sorted_couts_colonnes[1][j] - sorted_couts_colonnes[0][j] if sorted_couts_colonnes[:, j].shape[0] > 1 else sorted_couts_colonnes[0][j]

        max_penalite_ligne = penalites_lignes.max()
        max_penalite_colonne = penalites_colonnes.max()
        indexes_max_penalite_ligne = np.where(penalites_lignes == max_penalite_ligne)[0]
        indexes_max_penalite_colonne = np.where(penalites_colonnes == max_penalite_colonne)[0]

        if a:
            print("Ligne(s) de pénalité maximale :", indexes_max_penalite_ligne)
            print("Colonne(s) de pénalité maximale :", indexes_max_penalite_colonne)

        if len(indexes_max_penalite_ligne) > 1:
            max_penalite_ligne_index = 0
            for index in indexes_max_penalite_ligne:
                if min( provisions[index], commandes[np.argmin(couts[index])]) > max_penalite_ligne_index:
                    max_penalite_ligne_index = index
        else:
            max_penalite_ligne_index = indexes_max_penalite_ligne[0]

        if len(indexes_max_penalite_colonne) > 1:
            max_penalite_colonne_index = 0
            for index in indexes_max_penalite_colonne:
                if min(commandes[index],  provisions[np.argmin(couts[:, index])]) > max_penalite_colonne_index:
                    max_penalite_colonne_index = index
        else:
            max_penalite_colonne_index = indexes_max_penalite_colonne[0]

        selection = [0, 0]

        if max_penalite_ligne > max_penalite_colonne:
            selection[0] = max_penalite_ligne_index
            selection[1] = np.argmin(couts[max_penalite_ligne_index])
        elif max_penalite_colonne > max_penalite_ligne:
            selection[1] = max_penalite_colonne_index
            selection[0] = np.argmin(couts[:, max_penalite_colonne_index])
        else:
            if  provisions[max_penalite_ligne_index] > commandes[max_penalite_colonne_index]:
                selection[0] = max_penalite_ligne_index
                selection[1] = np.argmin(couts[max_penalite_ligne_index])
            else:
                selection[0] = np.argmin(couts[:, max_penalite_colonne_index])
                selection[1] = max_penalite_colonne_index

        if a:
            print("Arête à remplir :", selection)

        i_max, j_max = selection[0], selection[1]

        if  provisions[i_max] > commandes[j_max]:
            matrice_res[i_max, j_max] = commandes[j_max]
            couts[:, j_max] = 999999
        elif  provisions[i_max] < commandes[j_max]:
            matrice_res[i_max, j_max] =  provisions[i_max]
            couts[i_max, :] = 999999
        else:
            matrice_res[i_max, j_max] =  provisions[i_max]
            couts[i_max, :] = 999999
            couts[:, j_max] = 999999

        provisions[i_max] -= matrice_res[i_max, j_max]
        commandes[j_max] -= matrice_res[i_max, j_max]

        if a:
            print("Proposition à la fin de l'itération", iterations, ":")
            print(matrice_res)
    return(matrice_res)

def calcul_cout_total(matrice_cout,prop_transport):
    n=len(matrice_cout)
    m=len(matrice_cout[0])
    cout_tot=0
    for i in range(n):
        for j in range(m):
            cout_tot += matrice_cout[i][j] * prop_transport[i][j]

    return cout_tot

def connexe(transport):
    nb_ligne = len(transport)
    nb_colonne = len(transport[0])
    sommets = nb_ligne + nb_colonne

    adj_tab = [[] for _ in range(sommets)]
    for i in range(nb_ligne):
        for j in range(nb_colonne):
            if transport[i][j] != 0:
                adj_tab[i].append(nb_ligne + j)
                adj_tab[nb_ligne + j].append(i)

    return parcour_en_largeur_connexe(0,sommets,adj_tab)

def parcour_en_largeur_connexe(start,sommets,adj_tab):
    vu = [False] * sommets
    queue = [start]
    vu[start] = True

    while queue:
        sommet_actuel = queue.pop(0)
        for voisin in adj_tab[sommet_actuel]:
            if not vu[voisin]:
                vu[voisin] = True
                queue.append(voisin)

    return all(vu)




def acyclique(transport):
    nb_ligne = len(transport)
    nb_colonne = len(transport[0])
    sommets = nb_ligne + nb_colonne

    adj_tab = [[] for _ in range(sommets)]
    for i in range(nb_ligne):
        for j in range(nb_colonne):
            if transport[i][j] != 0:
                adj_tab[i].append(nb_ligne + j)
                adj_tab[nb_ligne + j].append(i)



    visited = [False] * sommets
    for i in range(sommets):
        if not visited[i]:
            a= parcour_en_largeur(i,sommets,adj_tab)
            if not a[0]:
                sommets_cycle=detect_cycle(sommets,adj_tab)
                sommets_ordonnes=ordonne_cycle(sommets_cycle,adj_tab)
                aff_cycle=affiche_cycle(sommets_ordonnes,nb_ligne)
                return [False,sommets_ordonnes,aff_cycle]

    return [True,0,0]

def detect_cycle(sommets,adj_tab):
    cycle=[]
    for i in range(sommets):
        a= parcour_en_largeur(i,sommets,adj_tab)
        if not a[0]:
            if a[1] not in cycle:
                cycle.append(a[1])
    return cycle

def ordonne_cycle(cycle,adj_tab):
    i=0
    index=0
    c=[cycle[0]]
    for j in adj_tab[c[0]]:
        if j in cycle  :
            c.append(j)
            break
    index+=1
    while i<len(cycle)-1:
        for j in adj_tab[c[index]]:
            if j in cycle and j!= c[index-1]:
                c.append(j)
                break
        index+=1
        i+=1
    return c

def affiche_cycle(sommets_ordonnes,nb_ligne):
    res=""
    for i in range(len(sommets_ordonnes)-1) :
        if sommets_ordonnes[i] < nb_ligne:
            res+="P"+str(sommets_ordonnes[i]+1)+" <==> "
        else:
            res+="C"+str(sommets_ordonnes[i]-nb_ligne+1)+" <==> "
    dernier_sommet=sommets_ordonnes[len(sommets_ordonnes)-1]
    if dernier_sommet< nb_ligne:
        res+="P"+str(dernier_sommet+1)
    else:
        res+="C"+str(dernier_sommet-nb_ligne+1)
    return res

def Maximisation(cycle,prop_trans,matrice_cout):
    coeff_delta=[0]*(len(cycle)-1)
    val=[0]*(len(cycle)-1)
    nb_ligne=len(prop_trans)
    nb_col=len(prop_trans[0])

    for i in range(len(cycle)-1):
        if i%2==0:
            coeff_delta[i]=1
        else:
            coeff_delta[i]=-1

    s=0
    delta=1
    for i in range(len(cycle)-1):
        if cycle[i]<nb_ligne:
            s+=coeff_delta[i]*matrice_cout[cycle[i]][cycle[i+1]-nb_ligne]
            val[i]=prop_trans[cycle[i]][cycle[i+1]-nb_ligne]
        elif cycle[i]>=nb_ligne:
            s+=coeff_delta[i]*matrice_cout[cycle[i+1]][cycle[i]-nb_ligne]
            val[i]=prop_trans[cycle[i+1]][cycle[i]-nb_ligne]
    if s>0:
        signe_delta=-1
    else:
        signe_delta=1

    delta=max(val)*signe_delta
    val_copy=[]
    for i in range(len(val)):
        val_copy.append(val[i]+(coeff_delta[i]*delta))

    min_val_copy=min(val_copy)

    if min_val_copy<0:
        delta=val[val_copy.index(min_val_copy)]*signe_delta

    for i in range(len(cycle)-1):
        if cycle[i]<nb_ligne:
            prop_trans[cycle[i]][cycle[i+1]-nb_ligne]=val[i]+(coeff_delta[i]*delta)
        else:
            prop_trans[cycle[i+1]][cycle[i]-nb_ligne]=val[i]+(coeff_delta[i]*delta)
    return prop_trans

def parcour_en_largeur(start,sommets,adj_tab):

    vu = [False] * sommets
    parents = [-1] * sommets

    queue = [start]
    vu[start] = True
    while queue:
        sommet_actuel = queue.pop(0)
        for voisin in adj_tab[sommet_actuel]:
            if not vu[voisin]:
                vu[voisin] = True
                parents[voisin] = sommet_actuel
                queue.append(voisin)
            elif parents[sommet_actuel] != voisin:
                return [False,voisin]
    return [True,[]]



def coutpotentiel(M, C):
    M=np.array(M)

    equations = []
    sommets =[]
    Lcolonne = list(M[:,0])
    nbzero = Lcolonne.count(0)
    colonne_index = 0
    for i in range(1, len(M[0])):
        if list(M[:, i]).count(0) < nbzero:
            Lcolonne = M[:, i]
            nbZero = list(Lcolonne).count(0)
            colonne_index = i

    nb_equat=0
    constantes=[]
    for i in range(len(M)):
      for j in range(len(M[0])):
        if M[i][j]!=0:
          nb_equat += 1
        if M[i][j]!=0:
          constantes.append(C[i][j])

    coefficients=np.zeros((nb_equat,len(M)+len(M[0])),dtype=int)
    k=0
    for i in range(len(M)):
      for j in range(len(M[0])):
        if M[i][j]!=0:
          coefficients[k][i]=1
          coefficients[k][len(M)+j]=-1
          k+=1

    eq=(len(M)+len(M[0]))*[0]
    eq[len(M)+colonne_index]=1
    coefficients=list(coefficients)
    coefficients.append(eq)
    constantes.append(0)



    coefficients_matrix=np.array(coefficients)
    constants_vector=np.array(constantes)


    solution = np.linalg.solve(coefficients_matrix, constants_vector)
    potentiel_prov=solution[:len(M):]
    potentiels_com=solution[len(M)::]


    matrice_cout_potentiel=np.zeros(((len(M)),len(M[0])),dtype=int)
    for i in range(len(M)):
        for j in range(len(M[0])):
           matrice_cout_potentiel[i,j]=potentiel_prov[i]-potentiels_com[j]

    return matrice_cout_potentiel

def coutpotentiel_prop_connexe(M, C,indices):

    equations = []
    sommets =[]
    Lcolonne = list(M[:,0])
    nbzero = Lcolonne.count(0)
    colonne_index = 0
    for i in range(1, len(M[0])):
        if list(M[:, i]).count(0) < nbzero:
            Lcolonne = M[:, i]
            nbZero = list(Lcolonne).count(0)
            colonne_index = i

    nb_equat=0
    constantes=[]
    for i in range(len(M)):
      for j in range(len(M[0])):
        if M[i][j]!=0:
            nb_equat += 1
            constantes.append(C[i][j])
        if i==indices[0] and j==indices[1]:
            nb_equat += 1
            constantes.append(C[i][j])

    coefficients=np.zeros((nb_equat,len(M)+len(M[0])),dtype=int)
    k=0
    for i in range(len(M)):
      for j in range(len(M[0])):
        if M[i][j]!=0:
            coefficients[k][i]=1
            coefficients[k][len(M)+j]=-1
            k+=1
        if i==indices[0] and j==indices[1]:
            coefficients[k][i]=1
            coefficients[k][len(M)+j]=-1
            k+=1

    eq=(len(M)+len(M[0]))*[0]
    eq[len(M)+colonne_index]=1
    coefficients=list(coefficients)
    coefficients.append(eq)
    constantes.append(0)



    coefficients_matrix=np.array(coefficients)
    constants_vector=np.array(constantes)


    solution = np.linalg.solve(coefficients_matrix, constants_vector)
    potentiel_prov=solution[:len(M):]
    potentiels_com=solution[len(M)::]


    matrice_cout_potentiel=np.zeros(((len(M)),len(M[0])),dtype=int)
    for i in range(len(M)):
        for j in range(len(M[0])):
           matrice_cout_potentiel[i,j]=potentiel_prov[i]-potentiels_com[j]

    return matrice_cout_potentiel

def calcul_cout_marginaux(couts_potentiels,matrice_cout):
    couts_marginaux=np.zeros((len(matrice_cout),len(matrice_cout[0])),dtype=int)
    for i in range(len(couts_marginaux)):
        for j in range(len(couts_marginaux[0])):
            couts_marginaux[i,j]=matrice_cout[i][j]-couts_potentiels[i][j]
    return couts_marginaux

def calcul_cout_marginaux(couts_potentiels,matrice_cout):
    couts_marginaux=np.zeros((len(matrice_cout),len(matrice_cout[0])),dtype=int)
    for i in range(len(couts_marginaux)):
        for j in range(len(couts_marginaux[0])):
            couts_marginaux[i,j]=matrice_cout[i][j]-couts_potentiels[i][j]
    return couts_marginaux



def prop_optimale(couts_marginaux):
    indicesVal =[]
    valeur_négative = couts_marginaux[0][0]
    for i in range (len(couts_marginaux)):
        for j in range(len(couts_marginaux[0])):
            if couts_marginaux[i][j] < valeur_négative:
                valeur_négative = couts_marginaux[i][j]
                indicesVal=[i,j]
    if valeur_négative<0:
        return [False,indicesVal]
    return [True,1]

def dict_cycle(cycle):
    d={}
    d[cycle[0]]=[cycle[len(cycle)-2],cycle[1]]
    for i in range(1,len(cycle)-1):
        d[cycle[i]]=[cycle[i-1],cycle[i+1]]
    return d

def marche_pied(prop_trans,M,iteration,non_connexe,prop_trans_non_connexe):
        print("*****Affichage des couts potentiels et couts marginaux*****")
        if non_connexe==True:
            potentiels=coutpotentiel(np.array(prop_trans_non_connexe),M)
        else:
            potentiels=coutpotentiel(prop_trans,M)
        couts_marginaux=calcul_cout_marginaux(potentiels,M)
        print("***Matrice couts potentiels***")
        afficher_matrice(potentiels)
        print()
        print("***Matrice couts marginaux***")
        afficher_matrice(couts_marginaux)
        print()
        optimale=prop_optimale(couts_marginaux)
        if optimale[0]==False:
            print("La proposition n'est pas optimale")
            arrete="P"+str(optimale[1][0]+1)+" <==> "+"C"+str(optimale[1][1]+1)
            print("L'arrete à ameliorer est",arrete)

            if non_connexe==False:
                prop_trans2=copy.deepcopy(prop_trans)
                prop_trans2[optimale[1][0]][optimale[1][1]]=1
                detect_cycle1=acyclique(prop_trans2)
                cycle=detect_cycle1[1]
                cycle_opt=[]
                cycle_opt.append(optimale[1][0])
                cycle_opt.append(optimale[1][1]+len(prop_trans))
                d=dict_cycle(cycle)
                while len(cycle_opt)!=len(cycle)-1:
                    i=cycle_opt[len(cycle_opt)-1]
                    if d[i][0] not in cycle_opt:
                        cycle_opt.append(d[i][0])
                    elif d[i][1] not in cycle_opt :
                        cycle_opt.append(d[i][1])
                cycle_opt.append(optimale[1][0])
                print("Le cycle créé est : ",affiche_cycle(cycle_opt,len(prop_trans)))
                print()
            else:
                prop_trans2=copy.deepcopy(prop_trans_non_connexe)
                prop_trans2[optimale[1][0]][optimale[1][1]]=1
                detect_cycle1=acyclique(prop_trans2)
                cycle=detect_cycle1[1]
                cycle_opt=[]
                cycle_opt.append(optimale[1][0])
                cycle_opt.append(optimale[1][1]+len(prop_trans))
                d=dict_cycle(cycle)
                while len(cycle_opt)!=len(cycle)-1:
                    i=cycle_opt[len(cycle_opt)-1]
                    if d[i][0] not in cycle_opt:
                        cycle_opt.append(d[i][0])
                    elif d[i][1] not in cycle_opt :
                        cycle_opt.append(d[i][1])
                cycle_opt.append(optimale[1][0])
                print("Le cycle créé est : ",affiche_cycle(cycle_opt,len(prop_trans)))
                print()

            print("Maximisation sur l arrete")
            prop_trans_opt=Maximisation(cycle_opt,prop_trans,M)
            print("***Nouvelle proposition***")
            afficher_matrice(prop_trans_opt)
            print()

            if connexe(prop_trans_opt) == True:
                print("******Nouvelle itération******")
                return marche_pied(prop_trans_opt,M,iteration+1,False,prop_trans_non_connexe)
            else:
                graphe=initialiser_graphe()
                graphe,p,q=Constructeur_graphe(prop_trans_opt, graphe)


                non_connexe_indices1,graphe1=ajout_connexite(graphe, prop_trans_opt,M)
                prop_trans_non_connexe=copy.deepcopy(prop_trans_opt)
                prop_trans_non_connexe[non_connexe_indices1[0]][non_connexe_indices1[1]]=1

                while connexe(prop_trans_non_connexe)==False:
                    non_connexe_indices,graphe1=ajout_connexite(graphe, prop_trans_non_connexe,M)
                    prop_trans_non_connexe[non_connexe_indices[0]][non_connexe_indices[1]]=1
                print("******Nouvelle itération******")
                return marche_pied(prop_trans_opt,M,iteration+1,True,prop_trans_non_connexe)
        else:
            print("La proposition est optimale")
            print("Fin marche pied")
            print("************************************************************")
            print()
            print("***Proposition de transport optimal***")
            afficher_matrice(prop_trans)
            print()
            cout_total=calcul_cout_total(M,prop_trans)
            print("Son cout total vaut",cout_total)
            return iteration

#Fonctions graphes

def initialiser_graphe():
    return {}

def ajouter_noeud(graphe, noeud):
    if noeud not in graphe:
        graphe[noeud] = []
    #print(graphe)

def ajouter_arete(graphe, source, destination):
    if source in graphe or destination in graphe:
        graphe[source].append(destination)
        #print(graphe)
    else:
        print("Les nœuds source ou destination n'existent pas dans le graphe.")

def afficher_graphe(graphe):
    for noeud in graphe:
        voisins = ",C".join(map(str, graphe[noeud]))
        print(f"P{noeud} --- C{voisins}")

def Constructeur_graphe(matrice_res, graphe):
 p=len(matrice_res)+len(matrice_res[0])
 q=0
 for i in range(len(matrice_res)):
    ajouter_noeud(graphe,i+1)
    for k in range(len(matrice_res[0])):
        if matrice_res[i][k]!=0:
            q+=1
            ajouter_arete(graphe, i+1, k+1)

 return graphe,p,q





#Si la connexité n'est pas verifié

def ajout_connexite(graphe, matrice_res,matrice_coût):

    liste=[]
    liste1=[]
    b=True

    for k in range(len(matrice_res)):
        for i in range(len(matrice_res[0])):
            if matrice_res[k][i]==0:
                liste.append([k,i])
                liste1.append(matrice_coût[k][i])

    coût_min=liste1[0]
    p=0
    while b==True :
        coût_min=liste1[0]
        p=0
        for k in range(0,len(liste1)):
            if coût_min>=liste1[k]:
                coût_min=liste1[k]
                p=k
        if liste[p][1]+1 not in graphe[liste[p][0]+1]:
                ajouter_arete(graphe, liste[p][0]+1 ,liste[p][1]+1 )
                matrice_res[liste[p][0]][liste[p][1]]-=1
        if acyclique(matrice_res)[0]==True:
                b=False
                matrice_res[liste[p][0]][liste[p][1]]+=1
                res=[liste[p][0],liste[p][1]]
                return res,graphe
        else:
                #print("Il y a un cycle en ajoutant l'arret:",liste[p] )
                matrice_res[liste[p][0]][liste[p][1]]+=1
                graphe[liste[p][0]+1].remove(liste[p][1]+1)
                liste.pop(p)
                liste1.pop(p)
                #print(liste,"et", liste1)
        if liste==[]:
            b=False
    return

def probleme_transport(a):
    print()
    if int(a) not in [1,2,3,4,5,6,7,8,9,10,11,12]:#verifie si a correspond à un numéro de table de test
        return "Au revoir."
    A="C:/Users/pc/Documents/RO fichiers test/"+str(a)+".txt"
    C,P,M=lectureRO(A)
    print("******************************Fichier test N°"+str(a)+"******************************")
    print("*Matrice de cout*")
    afficher_matrice(M)
    print()
    x=input("Quelle méthode souhaitez vous utiliser pour fixer la proposition initiale.Entrez B pour Ballas_Hammer et N pour la Nord_Ouest. ")
    print()
    meth=""
    if x=="B":
        prop_trans=Balas_Hammer(M,C,P,False)
        meth="Balas Hammer"
    else:
        prop_trans=Nord_Ouest(P,C)
        meth="Nord Ouest"

    #Marche pied
    print("Affichage de la proposition de transport avec la méthode "+meth)
    afficher_matrice(prop_trans)
    cout_prop_init=calcul_cout_total(M,prop_trans)
    print("Le cout total de la proposition de transport est :",cout_prop_init)
    print()

    print("************************************************************")
    print("Test pour savoir si la proposition de transport est dégénérée")
    print()
    detect_cycle=acyclique(prop_trans)
    if detect_cycle[0]==False:
        print("**************************************")
        print("Il y a un cycle : ",end="")
        print(detect_cycle[2])
        print("*****Destruction du cycle*****")
        prop_trans=Maximisation(detect_cycle[1],prop_trans,M)
        print("*****Affichage de la nouvelle proposition*****")
        print("**************************************")
    else:
        print("Il n'y a pas de cycle ")


    non_connexe=False
    if connexe(prop_trans)==True:
        print("La proposition est connexe")
        graphe=initialiser_graphe()
        graphe,p,q=Constructeur_graphe(prop_trans, graphe)
        print("Nombre de sommets",p)
        print("Nombre d'arrêts",q)
        afficher_graphe(graphe)
        non_connexe=False
        prop_trans_non_connexe=copy.deepcopy(prop_trans)
    else:
        prop_trans_non_connexe=copy.deepcopy(prop_trans)
        print("La proposition n'est pas connexe")
        graphe=initialiser_graphe()
        graphe,p,q=Constructeur_graphe(prop_trans, graphe)
        print("Nombre de sommets",p)
        print("Nombre d'arrêts",q)
        afficher_graphe(graphe)
        non_connexe_indices,graphe1=ajout_connexite(graphe, prop_trans,M)
        prop_trans_non_connexe[non_connexe_indices[0]][non_connexe_indices[1]]=1
        print()
        print("*****Fixation du problème*****")
        arrete_connexe="P"+str(non_connexe_indices[0]+1)+" <==> "+"C"+str(non_connexe_indices[1]+1)
        print("Arrete  ajouter : "+ arrete_connexe)
        graphe=initialiser_graphe()
        graphe,p,q=Constructeur_graphe(prop_trans_non_connexe, graphe)
        print("Nombre de sommets",p)
        print("Nombre d'arrêts",q)
        print("Le graphe devient :")
        afficher_graphe(graphe1)
        print()
        while connexe(prop_trans_non_connexe)==False:
            non_connexe_indices,graphe1=ajout_connexite(graphe, prop_trans_non_connexe,M)
            prop_trans_non_connexe[non_connexe_indices[0]][non_connexe_indices[1]]=1
            arrete_connexe="P"+str(non_connexe_indices[0]+1)+" <==> "+"C"+str(non_connexe_indices[1]+1)
            graphe=initialiser_graphe()
            graphe,p,q=Constructeur_graphe(prop_trans_non_connexe, graphe)
            print("Arrete  ajouter : "+ arrete_connexe)
            print("Nombre de sommets",p)
            print("Nombre d'arrêts",q)
            print("Le graphe devient :")
            afficher_graphe(graphe1)
            print()
        non_connexe=True

    print()
    print("Fin du test")
    print("************************************************************")
    print()
    print("************************************************************")
    print("Déclenchement du marche pied")
    nb_iteration=marche_pied(prop_trans,M,0,non_connexe,prop_trans_non_connexe)
    if nb_iteration>0:
        print("Contre "+str(cout_prop_init)+" pour la proposition intitial avec la méthode "+meth)
        print()
        print("La proposition optimale à été obtenu après "+str(nb_iteration)+" itérations.")
    print()
    print("******************************Fin test fichier N°"+str(a)+"******************************")
    print()

    x=input("Entrez un numéro de fichier si vous souhaitez continuer.Tapez 0 sinon ")
    if int(x) not in [1,2,3,4,5,6,7,8,9,10,11,12]:
        return "Au revoir."
    return probleme_transport(x)


