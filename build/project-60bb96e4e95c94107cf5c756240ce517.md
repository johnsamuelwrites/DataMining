# Project

## Goals

- Implementation of a well-commented [Recommender system](https://en.wikipedia.org/wiki/Recommender_system) in Python.
  - Automate image data collection and annotation.  
  - Analyze data to identify trends.
  - Visualize analysis results effectively.  
  - Build and validate a recommendation system.
  - Ensure thorough testing of all components.  
- Present findings in a detailed report.  


The goal of this project is to recommend images based on the preferences
of the user. You have three practical sessions to build this system. You
must ensure that all the tasks related to data acquisition, annotation,
analysis, and visualization are automated.

The main tasks of the project are given below:

1.  Data Collection
2.  Labeling and Annotation
3.  Data Analyses
4.  Data Visualization
5.  Recommendation System
6.  Tests
7.  Report

![Architecture](../../images/Project-Architecture.png "Architecture")

## Data Collection

You have to collect and download a set of images. You have the following
tasks to program, automating the process as much as possible:

1.  Create a folder called *images*.
2.  Download open-licensed images to the folder *images* (minimum 100
    images).
3.  Save metadata of every image like image size, image format (.jpeg,
    .png, etc.), image orientation (landscape, portrait, square, etc.),
    creation date, camera model, etc. in one or more JSON files. You can
    make use of the [Exif](https://en.wikipedia.org/wiki/Exif)
    information present in the image files.

For this task, you should look for sources having additional information
like the tags, categories, etc.

## Labeling and Annotation

In this task, you may need to label, annotate and save additional
information about every image. You may analyze the images using
clustering algorithms for finding the predominant colours.

You already have some metadata from the EXIF of images from the previous
task. In this task, your goal is to obtain additional information, like
the predominant colors, tags. How about asking users to tag the images?
E.g., color names, \#cat, \#flower, \#sunflower, rose etc. How are you
planning to process the user tags? Is it possible to automate this
process?

## Data Analyses

Ask the user to select some images and add tags. For every user, you are
now ready to build a user-preference profile, based on this selection.
You may collect the following information manually, but the objective of
this task is to obtain them using the selected images in an automated
manner:

1.  Favorite colors
2.  Favorite image orientation
3.  Favorite image sizes (thumbnail images, large images, medium-size
    images, etc.)
4.  Favorite tags
5.  \...

Now, with your knowledge of different types of classifiers and
clustering algorithms, what more information will you add for every
image?

Your next objective is to analyze the user information and their
favorite images. How did you create random users? How many users did you
create? What information did you store for every user? What types of
analyses did you perform?

## Data Visualization

In this task, your goal is to visualize the different characteristics of
all the downloaded images.

1.  The available number of images for every year
2.  The available number of images for different types: image size,
    image orientation, camera models, etc.
3.  Color characteristics

The users may also like to visualize the above information related to
their favorite images. In this task, you must also add functionality to
let the users visualize information related to their own user profile.

## Recommendation System

Are you now ready to recommend images to a user? In this task, your goal
is to build the recommendation system. Which approach did you decide to
take? Collaborative filtering, content-based, or a hybrid approach? Which
algorithm(s) did you choose (classifcation, clustering,...) ? For
every user, are you now in a position to build a user-preference
profile? What type of information did you use for building a user
profile? What\'s missing? What are the limitations of your proposed
approach?

## Tests

Your next task is to develop and run different tests on your proposed
system. Are different functions functional? How did you test your
project? How are you verifying that your recommender system is working?

## Report

Your final task is to prepare a 4-page Project report (French or
English) in PDF format detailing the following:

-   The goal of your project
-   Data sources of your images and license.
-   Size of your data.
-   Information that you decided to store for each image.
-   Information concerning user preferences
-   Data mining and/or machine learning models that you used along with
    the metrics obtained.
-   Self-evaluation of your work.
-   Remarks concerning the practical sessions, exercises, and scope for
    improvement.
-   Conclusion

**Note**: Please do not add any program (or code) in this report.


## Submission


-   Please **do not** submit your images.
-   Rename your project report as Name1\_Name2\_\[Name3\].pdf, where
    Name1, Name2, etc. are your names.
-   Add your project report in your project folder.
-   Compress and rename your project work as
    Name1\_Name2\_\[Name3\].zip, where Name1, Name2 are your names.
-   Submit your **project work** online.


### Evaluation


The criteria for the project evaluation is given below:

1.  Data Collection
    1. Automated approaches to data collection
    2. Use of open-licensed images
    3. Storage and management of images and the associated metadata
2.  Labeling and Annotation
    1. Automated approaches to labeling
    2. Storage and management of labels and annotations of images
    3. Use of classification and clustering algorithms
3.  Data Analyses
    1. Types of analyses used
    2. Use of Pandas and Scikit-learn
    3. Use of data mining algorithms
4.  Data Visualization
    1. Types of visualization techniques used
    2. Use of matplotlib
5.  Recommendation System
    1. Storage and management of user preferences and user-profile
    2. Use of recommendation algorithms
6.  Tests
    1. Presence of functional tests
    2. Presence of user tests
7.  Report
    1. Clarity of presentation
    2. Presence of a clear introduction and conclusion, architecture
        diagrams, a summary of different tasks achieved, and limitations
    3. Bibliography

**Note**: You can check [supplementary examples](../../examples) of noteooks.
