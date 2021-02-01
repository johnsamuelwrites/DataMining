### Objectifs

-   Comprendre les bases de la programmation Python :
-   Variables, types de données, listes, ensembles et tuples.
-   Expressions conditionnelles et boucles
-   Trier
-   Dictionnaire
-   Fichiers et interaction avec les utilisateurs



#### Exercice 1

1\. Commentaires



``` 
# Ceci est un commentaire
print("Bonjour")
```



2\. Variables



``` 
# une variable
message = "le monde!"
print(message)
```



``` 
a = 10
b = 20
c = a + b
print(c)
```



``` 
# nombres réels
pi = 3.14
print(pi)
```



``` 
# types de donnés
message1 = "Bonjour"
a = 12
pi = 3.14
print(type(message1))
print(type(a))
print(type(pi))
```



3\. Concaténation de deux chaînes de caractères



``` 
# Concaténation de deux chaînes de caractères
message = "le monde!"
print("Bonjour" + message)
```



``` 
# Concaténation de deux chaînes de caractères
message1 = "Bonjour "
message2 = "le monde!"
print(message1 + message2)
```



``` 
# concaténation de deux variables de types de données différents
# opération + sur deux types de données différents

# Supprimer le commentaire ci-dessous et lancer le code
message1 = "Bonjour en Python"
a = 3
#print(message1 + a)
```



Pourquoi cette erreur ? Dans le code suivant, nous corrigeons cette erreur.



``` 
# solution de concaténation entre deux variables de types de données différents
message1 = "Bonjour en Python "
a = 3
print(message1 + str(a))
```



4\. Listes



``` 
a = [10, 20, 30, 40, 50]
print(a)
```



``` 
a = [10, 20, 30, 40, 50]
print(a[0])
print(a[1])
print(a[2])
print(a[3])
print(a[4])
```



``` 
#Supprimer le commentaire ci-dessous et lancer le code
a = [10, 20, 30, 40, 50]
#print(a[8])
```



Nous essayons d\'accéder à un élément à un index qui n\'existe pas



``` 
message1 = "Bonjour en Python "
print(message1[0])
print(message1[1])
print(message1[2])
print(message1[3])
print(message1[4])
print(message1[5])
print(message1[6])
print(message1[7])
```



Le code ci-dessus affiche les différents caractères de la chaîne (ou liste de caractères).
Nous allons maintenant obtenir la longueur de cette chaîne.



``` 
message1 = "Bonjour en Python "
print(len(message1))
```



Nous allons maintenant créer une liste d'entiers.



``` 
a = [10, 20, 30, 40, 50]
print(len(a))
```



``` 
a = [10, 20, 30, 40, 50]

# ajouter un nouveau numéro à la fin de la liste
a.append(60)
print(a)
```



``` 
a = [10, 20, 30, 40, 50]

# modifier un numéro à un index donné
a[0] = 0
print(a)
```



``` 
#Supprimer le commentaire ci-dessous et lancer le code
a = [10, 20, 30, 40, 50]

#a[6] = 20
print(a)
```



Pourquoi cette erreur? Nous modifions un élément à un indice inexistant.



``` 
a = [10, 20, 30, 40, 50]

# l'insertion d'un élément à un index particulier modifiera la liste
a.insert(0, 0)
print(a)
print(len(a))
```



``` 
a = [10, 20, 30, 40, 50]
a.insert(6,60)
print(a)
print(len(a))
```



``` 
a = [10, 20, 30, 40, 50]

# Nous allons maintenant essayer d'insérer un nombre à un index plus grand que la longueur
# de la liste. Nous veillerons à ce qu'il n'y ait pas d'erreur et le nouveau numéro
# est ajouté à la fin de la liste
a.insert(10,60)
print(a)
print(len(a))
```



5\. Tuples (non-modifiable lists)



``` 
a = (10, 20, 30, 40, 50)
print(a)
```



``` 
a = (10, 20, 30, 40, 50)
print(a[0])
```



``` 
a = (10, 20, 30, 40, 50)


# Nous essayons maintenant de modifier un tuple
# Décommentez le code ci-dessous et lancez le code
# Un tupe est une liste non modifiable

#a[0] = 0
print(a)
```



6\. Sets



``` 
# Un ensemble(Set) est une collection d'éléments distincts

a = {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}
print(a)
```



``` 
a = {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}
a.add(10)
print(a)
```



``` 
a = {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}
a.add(60)
print(a)
```



``` 
a = {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}
a.remove(40)
print(a)
```



Nous allons maintenant essayer différents types de données avec les chiffres et afficher le résultat.



``` 
# set
a = {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}
print(a)
print(type(a))
 
# tuple
b = (10, 20, 30, 40, 50, 10, 20, 30, 40, 50)
print(b)
print(type(b))
 
# list
c= [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
print(c)
print(type(c))
```



#### Exercice 2

1\. Expressions conditionnelles



``` 
a = 12
if( a%2 == 0):
    print(a, " is divisible by 2")
else:
    print(a, " is not divisible by 2")
```



``` 
lang = "Français"
if (lang =="Français"):
    print("Bonjour le monde!")
else:
    print("Hello World!")
```



2\. Boucles

Les boucles peuvent également être utilisées pour accéder aux éléments de différents indices.



``` 
# liste
for i in [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]:
    print(i)
```



``` 
# tuples
for i in (10, 20, 30, 40, 50, 10, 20, 30, 40, 50):
    print(i)
```



``` 
# set
for i in {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}:
    print(i)
```



2\. Limites



``` 
for i in range(0,10):
    print(i)
```



``` 
for i in range(0,10,2):
    print(i)
```



``` 
# print() par défaut affiche le message suivi d'une nouvelle ligne
# Mais vous pouvez changer son comportement

for i in range(0,10,2):
    print(i, end=' ')
```



``` 
for i in range(10,0,-2):
    print(i)
```



``` 
for i in range(10,0):
    print(i)
```



split(): La fonction peut être utilisée pour séparer une chaîne de caractères en utilisant
un délimiteur spécifié. Par défaut, le délimiteur est un espace blanc.



``` 
for i in "Bonjour,le,monde!".split():
    print(i)
```



``` 
for i in "Bonjour,le,monde!".split(","):
    print(i)
```



Write a program in Python to display the following output

1

12

123

1234

12345

123456

1234567

12345678



#### Exercice 3

1\. Tri



``` 
num = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
num.sort()
print(num)
```



2\. Tri (ordre décroissant)



``` 
num = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
num.sort(reverse=True)
print(num)
```



3\. minimum



``` 
num = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
print(min(num))
```



4\. maximum



``` 
num = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
print(max(num))
```



5\. sorted()

Vous pouvez utiliser cette fonction si vous ne souhaitez pas modifier votre liste initiale par un tri.



``` 
num = [70, 20, 30, 10, 50, 60, 20, 80, 70, 50]
tri = sorted(num,reverse=True)
print(num)
print(tri)
```



``` 
num = [70, 20, 30, 10, 50, 60, 20, 80, 70, 50]

# sélectionner les cinq premiers numéros
tri = sorted(num,reverse=True)[:5]
print(tri)
```



Modifiez le code donné ci-dessous pour afficher les cinq plus grands numéros uniques.



``` 
print(sorted("Bonjour le monde!".split(), key=str.lower, 
           reverse=True))
  


```



#### Exercice 4

1\. Dictionnaire



``` 
a = {"contente": 12, "content": 12, "triste": 2}
print(a)
print(type(a))
```



``` 
a = {"contente": 12, "content": 12, "triste": 2}
for key in a:
    print("la phrase ", key, " apparait ", a[key], " fois")
```



``` 
a = {"contente": 12, "content": 12, "triste": 2}
for key,value in a.items():
    print("la phrase ", key, " apparait ", value, " fois")
  


```



``` 
a = {"contente": 12, "content": 12, "triste": 2}
a["joie"] = 10
print(a)
  


```



``` 
a = {"contente": 12, "content": 12, "triste": 2}
del a["triste"]
print(a)
```



``` 
mots = {"contente": 12, "content": 12, "triste": 2, 
     "joie" : 10}
print(sorted(mots))
```



``` 
mots = {"contente": 12, "content": 12, "triste": 2, 
     "joie" : 10}
mots_tuple = [(key, value) for key,value in mots.items()]
print(mots_tuple)
```



2\. itemgetter



``` 
from operator import itemgetter
 
mots = {"contente": 12, "content": 12, "triste": 2, 
     "joie" : 10}
mots_tuple = [(key, value) for key,value in mots.items()]
print(sorted(mots_tuple, key=itemgetter(1)))
```



3\. Interaction with user



``` 
nom = input("Quel est votre nom?")
print(nom)
```



``` 
age = input("Quel est votre âge? ")
print(age)
print(type(age))
```



``` 
age = input("Quel est votre âge? ")
age = int(age)
print(age)
print(type(age))
```



**Question:** Ecrire un programme en Python qui interagit avec l\'utilisateur pour
obtenir les informations suivantes de 5 étudiants :

-   Nom de l\'étudiant
-   Âge
-   Grades en 5 modules

Une fois les informations obtenues pour les cinq étudiants, calculez
et afficher les valeurs suivantes pour chaque module :

-   note moyenne
-   note maximale
-   note minimale



#### Exercice 5

Lire et écrire dans un fichier



1\. Fichiers



``` 
# écrire dans un fichier
message = "Bonjour le monde"
with open("bonjour.txt", "w") as file:
    file.write(message)
file.close()
```



``` 
# lire un fichier
with open("bonjour.txt", "r") as file:
    text = file.read()
    print(text)
file.close()
```



``` 
# écrire dans un fichier
message1 = "Bonjour le monde"
message2 = "Programmation en Python"
with open("bonjour.txt", "w") as file:
    file.write(message1)
    file.write(message2)
file.close()
```



``` 
# lire un fichier
with open("bonjour.txt", "r") as file:
    text = file.read()
    print(text)
file.close()
```



``` 
# Rédaction dans un fichier en utilisant le caractère de nouvelle ligne
message1 = "Bonjour le monde\n"
message2 = "Programmation en Python"
with open("bonjour.txt", "w") as file:
    file.write(message1)
    file.write(message2)
file.close()
 
with open("bonjour.txt", "r") as file:
    text = file.read()
    print(text)
file.close()
```



2\. readline()

Cette fonction peut être utilisée pour lire un fichier ligne par ligne et non le contenu complet en un seul appel comme read()



``` 
message1 = "Bonjour le monde\n"
message2 = "Programmation en Python"
with open("bonjour.txt", "w") as file:
    file.write(message1)
    file.write(message2)
file.close()
```



``` 
with open("bonjour.txt", "r") as file:
    text = file.readline()
    print(text)
file.close()
```



``` 
message1 = "Bonjour le monde\n"
message2 = "Programmation en Python\n"
with open("bonjour.txt", "w") as file:
    file.write(message1)
    file.write(message2)
file.close()
```



``` 
with open("bonjour.txt", "r") as file:
    for line in file:
        print(line)
file.close()
```



**Question:** Copiez un fichier HTML dans votre répertoire. Ecrire un programme
en Python pour obtenir les valeurs suivantes :

-   nombre de caractères dans le fichier HTML
-   nombre de lignes dans le fichier HTML
-   nombre de mots dans le fichier HTML
-   les vingt premiers mots du fichier HTML
-   des mots distincts dans le fichier HTML



**Question:** Copier le CSV
du fichier population.csv du dossier **data**. Le fichier contient les valeurs de population entre
1901 et 2016. Écrivez un programme en Python pour obtenir la valeur maximale.

-   la valeur maximale de la population
-   la valeur minimale de la population



``` 
```

