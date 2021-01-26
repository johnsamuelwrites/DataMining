### Goals

* Understanding fundamentals of Python programming:
* Variables, data types, lists, sets and tuples.
* Conditional expressions and loops
* Sort
* Dictionary
* Files and user interaction

#### Exercise 1


1\. Comments

``` 
 # This is a comment
           print("Bonjour")
```

2\. Variables

``` 
            # a variable
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
            
# floating point numbers
pi = 3.14
print(pi)
            
          
```

``` 
            
# data types
          message1 = "Bonjour"
a = 12
pi = 3.14
print(type(message1))
print(type(a))
print(type(pi))
            
          
```

3\. Concatenation of two strings

``` 
            
            # concatenation of two strings
        message = "le monde!"
        print("Bonjour" + message)
            
          
```

``` 
            
# concatenation of two strings
          message1 = "Bonjour "
          message2 = "le monde!"
print(message1 + message2)
            
          
```

``` 
            
# concatenation involving two variables of different data types
# operation + on two different data types
          message1 = "Bonjour en Python"
a = 3
print(message1 + a)
            
          
```

``` 
            
# concatenation solution involving two variables of different data types
          message1 = "Bonjour en Python "
a = 3
print(message1 + str(a))
            
          
```

4\. Lists

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
            
a = [10, 20, 30, 40, 50]
print(a[8])
            
          
```

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

``` 
            
message1 = "Bonjour en Python "
print(len(message1))
            
          
```

``` 
            
a = [10, 20, 30, 40, 50]
print(len(a))
            
          
```

``` 
            
a = [10, 20, 30, 40, 50]
a.append(60)
print(a)
            
          
```

``` 
            
a = [10, 20, 30, 40, 50]
a[0] = 0
print(a)
            
          
```

``` 
            
a = [10, 20, 30, 40, 50]
a[6] = 20
print(a)
            
          
```

``` 
            
a = [10, 20, 30, 40, 50]
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
a[0] = 0
print(a)
            
          
```

6\. Sets

``` 
            
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


#### Exercise 2


1\. Conditional Expressions

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

2\. Loops

``` 
            
for i in [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]:
  print(i)
            
          
```

``` 
            
for i in (10, 20, 30, 40, 50, 10, 20, 30, 40, 50):
  print(i)
            
          
```

``` 
            
for i in {10, 20, 30, 40, 50, 10, 20, 30, 40, 50}:
  print(i)
            
          
```

2\. Range

``` 
            
for i in range(0,10):
  print(i)
            
          
```

``` 
            
for i in range(0,10,2):
  print(i)
            
          
```

``` 
            
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

``` 
            
for i in "Bonjour,le,monde!".split():
  print(i)
            
          
```

``` 
            
          for i in "Bonjour,le,monde!".split(","):
  print(i)
            
          
```

Write a program in Python to display the following output

``` 
            
1
12
123
1234
12345
123456
1234567
12345678
            
          
```


#### Exercise 3


1\. Sort

``` 
            
num = [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]
num.sort()
print(num)
            
          
```

2\. Sort (decreasing order)

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

``` 
            
num = [70, 20, 30, 10, 50, 60, 20, 80, 70, 50]
num = sorted(num,reverse=True)[:5]
print(num)
            
          
```

Modify the code given below to display the five greatest unique numbers.

``` 
            
print(sorted("Bonjour le monde!".split(), key=str.lower, 
          reverse=True))
            
          
```


#### Exercise 4


1\. Dictionary

``` 
            
a = {"contente": 12, "content": 12, "triste": 2}
print(a)
print(type(a))
            
          
```

``` 
            
          a = {"contente": 12, "content": 12, "triste": 2}
for cle in a:
            print("la phrase ", key, " apparait ", a[cle], " fois")
            
          
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

3\. Interaction avec l\'utilisateur

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

**Question:** Write a program in Python that interacts with the user to
obtain the following information of 5 students:

-   Name of student
-   Age
-   Grades in 5 modules

Once the information for all the five students is obtained, calculate
and display the following values for every module:

-   average grade
-   maximum grade
-   minimum grade


#### Exercise 5


1\. Files

``` 
            
          message = "Bonjour le monde"
with open("bonjour.txt", "w") as file:
  file.write(message)
file.close()
            
          
```

``` 
            
          with open("bonjour.txt", "r") as file:
  text = file.read()
  print(text)
file.close()
            
          
```

``` 
            
          message1 = "Bonjour le monde"
          message2 = "Programmation en Python"
          with open("bonjour.txt", "w") as file:
  file.write(message1)
  file.write(message2)
file.close()
            
          
```

``` 
            
          with open("bonjour.txt", "r") as file:
  text = file.read()
  print(text)
file.close()
            
          
```

``` 
            
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


**Question:** Copy any HTML file in your home directory. Write a program
in Python to get the following values:

-   number of characters in the HTML file
-   number of lines in the HTML file
-   number of words in the HTML file
-   first twenty words in the HTML file
-   distinct words in the HTML file

**Question:** Copy [CSV
file](../../../../../en/teaching/courses/2017/DataMining/population.csv)
in your home directory. The file contains the population values between
1901 and 2016. Write a program in Python to get the maximum value.

-   the maximum value of population
-   the minimum value of population


#### References

[Link](../references.md)
