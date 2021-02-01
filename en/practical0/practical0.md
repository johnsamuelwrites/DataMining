### Goals

-   Understanding fundamentals of Python programming:
-   Variables, data types, lists, sets and tuples.
-   Conditional expressions and loops
-   Sort
-   Dictionary
-   Files and user interaction



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

#Uncomment the print statement and run the code
message1 = "Bonjour en Python"
a = 3
#print(message1 + a)
```



Why did you get this error? In the following code, we correct this error.



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
#Uncomment the print statement and run the code
a = [10, 20, 30, 40, 50]
#print(a[8])
```



Why did you get this error? We are trying to access a element at an index that does not exist.



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



The above code displayed the individual characters in the string (or list of characters).
We will now get the length of this string.



``` 
message1 = "Bonjour en Python "
print(len(message1))
```



We are now goint to create a list of integers.



``` 
a = [10, 20, 30, 40, 50]
print(len(a))
```



``` 
a = [10, 20, 30, 40, 50]

# add a new number at the end of the list
a.append(60)
print(a)
```



``` 
a = [10, 20, 30, 40, 50]

# modify a number at a particular index
a[0] = 0
print(a)
```



Why did we get this error? We are modifying an element at a non-existing index.



``` 
#Uncomment the assignment statement and run the code
a = [10, 20, 30, 40, 50]
#a[6] = 20
print(a)
```



``` 
a = [10, 20, 30, 40, 50]

# inserting an element at a particular index will modify the list
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

# We will now try to insert a number at an index greater than the length
# of the list. We will see that we do not get any error and the new number
# is added at the end of the list
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

# We now try to modify a tuple
# Uncomment the code below and run the code
# A tupe is a non-modifiable list

#a[0] = 0
print(a)
```



6\. Sets



``` 
# A set is a collection of distinct elements
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



We will now try different data types with the numbers and print the result



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

Loops can also be used to access the elements at different indices.



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
# print() by default displays the message followed by a new line
# But you can change its behaviour
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



split(): the function can be used to separate a string using a specified delimited.
By default, the delimiter is a white space.



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

You can use the function if you do not wish to modify your initial list by sorting.



``` 
num = [70, 20, 30, 10, 50, 60, 20, 80, 70, 50]
sortednum = sorted(num,reverse=True)
print(num)
print(sortednum)
```



``` 
num = [70, 20, 30, 10, 50, 60, 20, 80, 70, 50]

# select first five numbers
sortednum = sorted(num,reverse=True)[:5]
print(sortednum)
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

This function can be used to read a file line by line and not the complete content in a single call like read()



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



**Question:** Copy the CSV
file population.csv from the **data** folder. The file contains the population values between
1901 and 2016. Write a program in Python to get the maximum value.

-   the maximum value of population
-   the minimum value of population



``` 
```

