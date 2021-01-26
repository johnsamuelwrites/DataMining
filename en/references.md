### Tools and libraries

-   [Jupyter](https://jupyter.org/)
-   [Scikit-learn](http://scikit-learn.org/stable/index.html)
-   [NumPy](http://www.numpy.org/)
-   [Pandas](https://pandas.pydata.org/)
-   [urllib.request](https://docs.python.org/3.6/library/urllib.request.html)
-   [matplotlib.pyplot](https://matplotlib.org/api/pyplot_api.html)
-   [Tensorflow](https://www.tensorflow.org/)

### Tutorials and Manual pages

-   Quick start tutorials
    -   [NumPy QuickStart tutorial](https://numpy.org/doc/stable/user/quickstart.html)
    -   [Pandas 10min tutorial](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
    -   [Scikit tutorial](http://scikit-learn.org/stable/tutorial/basic/tutorial.html#introduction)
    -   [Tensorflow: Getting Started for ML Beginners](https://www.tensorflow.org/get_started/get_started_for_beginners)
-   Visualization
    -   [Pie and polar charts](https://matplotlib.org/examples/pie_and_polar_charts/pie_demo_features.html)
-   Clustering
    -   [OpenCV and Python K-Means Color Clustering](https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/)
    -   [Scikit-learn Clustering](http://scikit-learn.org/stable/modules/clustering.html)
    -   [K-Means algorithm](http://scikit-learn.org/stable/modules/clustering.html#k-means)
    -   [Mini Batch K-Means algorithm](http://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans)
-   Classifier
    -   [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
    -   [DNNClassifier](https://web.archive.org/web/20190602073143/https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
    -   [Classifier comparison](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py)
-   Regression
    -   [Linear model](http://scikit-learn.org/stable/modules/linear_model.html)
    -   [Linear regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    -   [Polynomial Features](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
    -   [Polynomial interpolation](http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py)
-   Neural networks
    -   [Neural network models (supervised)](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
    -   [Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
    -   [Multi-layer Perceptron](http://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron)
    -   [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
-   JSON
    -   [JSON encoder and decoder](https://docs.python.org/3.6/library/json.html)

### Python

#### Resize images or create thumbnail images

``` 
              from PIL import Image
              import matplotlib.pyplot as plot
              
              imgfile = Image.open("flower.jpg")
              #resize image to 200px x 200 px using ANTIALIAS interpolation
              imgfile = imgfile.resize([200, 200], Image.ANTIALIAS)
              
              plot.imshow(imgfile)
              plot.show()
        
```

#### Create thumbnail images

``` 
              from PIL import Image
              import matplotlib.pyplot as plot
              
              imgfile = Image.open("flower.jpg")
              imgfile.thumbnail([200, 200], Image.ANTIALIAS)
              
              plot.imshow(imgfile)
              plot.show()
          
```

#### Create JSON data

``` 
              import json
              
              colors = [[120, 30, 120], [150, 30, 10]]
              user = "user 1"
              
              data = dict()
              data['user'] = user
              data['colors'] = colors
              
              print(json.dumps(data))
```

#### Write JSON data to file

``` 
              import json
              
              colors = [[120, 30, 120], [150, 30, 10]]
              user = "user 1"
              
              data = dict()
              data['user'] = user
              data['colors'] = colors
              
              with open("user.json", "w") as file:
                  json.dump(data, file)
              
              file.close()
          
```

#### Read JSON data from file

``` 
              import json
              
              data = ""
              with open("user.json", "r") as file:
                  data = json.load(file)
              
              print(data)
        
```

### SPARQL queries

#### Get names of 100 programming languages.

``` 
 
          
            SELECT ?languageLabel (YEAR(?inception) as ?year)
            WHERE
            {
             #instances of programming language
             ?language wdt:P31 wd:Q9143;
              wdt:P571 ?inception;
              rdfs:label ?languageLabel.
             FILTER(lang(?languageLabel) = "en")
            }
            ORDER BY ?year
            LIMIT 100
          
        
```

#### Get names of 100 programming languages and their paradigms.

``` 
 
          
            SELECT ?languageLabel ?paradigmLabel (YEAR(?inception) as ?year)
            WHERE
            {
             #instances of programming language
             ?language wdt:P31 wd:Q9143;
              wdt:P571 ?inception; #inception
              wdt:P3966 ?paradigm; #programming language paradigm       
              rdfs:label ?languageLabel. #label
             ?paradigm rdfs:label ?paradigmLabel #label
             FILTER(lang(?languageLabel) = "en" && lang(?paradigmLabel) = "en") #English
            }
            ORDER BY ?year ?paradigmLabel 
            LIMIT 100
          
        
```

#### Get available information of population of different countries at different periods of time.

``` 
 
          
            SELECT DISTINCT ?countryLabel (YEAR(?date) as ?year) ?population
            WHERE {
             ?country wdt:P31 wd:Q6256; #Country 
               p:P1082 ?populationStatement;
              rdfs:label ?countryLabel. #Label
             ?populationStatement ps:P1082 ?population; #population
              pq:P585 ?date. #period in time
             FILTER(lang(?countryLabel)="en") #Label in English
            }
            ORDER by ?countryLabel ?year
            LIMIT 1000
          
        
```

#### Get population of France at different periods of time.

``` 
 
          
             SELECT  (YEAR(?date) as ?year)?population
             WHERE
             {
               VALUES ?country {wd:Q142}
               ?country p:P1082 ?populationStatement.
               ?populationStatement ps:P1082 ?population;
                                    pq:P585 ?date.
             }
             ORDER by ?year
          
        
```

#### Get number of programming languages released every year.

``` 
 
          
           SELECT ?year (COUNT(?programmingLanguage) as ?count)
           WHERE
           {
             ?programmingLanguage wdt:P31/wdt:P279* wd:Q9143;
               wdt:P571 ?date.
             BIND(YEAR(?date) as ?year)
           }
           GROUP BY ?year 
           ORDER by ?year
          
        
```

#### Get number of programming languages released every year belonging to different paradigms.

``` 
 
          
           SELECT ?year ?paradigmLabel (COUNT(?programmingLanguage) as ?count)
           WHERE
           {
             ?programmingLanguage wdt:P31/wdt:P279* wd:Q9143;
               wdt:P3966 ?paradigm;
               wdt:P571 ?date.
             ?paradigm rdfs:label ?paradigmLabel FILTER(lang(?paradigmLabel)="en").
             BIND(YEAR(?date) as ?year)
           }
           GROUP BY ?year ?paradigmLabel
           ORDER by ?year ?paradigmLabel
          
        
```

#### Get number of softwares released every year.

``` 
 
          
             SELECT ?year (COUNT(?software) as ?count)
             WHERE
             {
               ?software wdt:P31/wdt:P279* wd:Q7397;
                 wdt:P571 ?date.
               BIND(YEAR(?date) as ?year)
             }
             GROUP BY ?year
             ORDER by ?year
          
        
```

### Standards and tools

-   [W3C Math](https://www.w3.org/Math/)
-   [Color Tool-Material design](https://material.io/color/)


