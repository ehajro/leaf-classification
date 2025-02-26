## Dataset

The dataset that I have used for this project can be found here: https://www.dropbox.com/scl/fi/ugaki2x9bhj14bfnue0mb/LeafData.mat?rlkey=3uftqdf8g2jhri3qzq7srdpbz&st=ayzp1aww&dl=0

## Overview

In this project, I am working with a data file that contains pictures of leaves that belong to different tree species. More specifically, the dataset has 4 variables: 
Ptrain, Ptest, names, and Prgb. The "Ptrain" variable contains grey scale images of leaves from 10 different tree species, and each species has 7 different pictures in it. 
We can access each of these pictures by indexing Ptrain{i}{j}, where i takes values from 1 to 10 and represents the different species, and j takes values from 1 to 7 and represents 
each different picture that belongs to species i. The next variable, "Ptest", also contains images from 10 different species, but this time, we only have 3 images for each species. 
We can access each image in a similar way to Ptrain, but here our j index only takes values from 1 to 3. Each image in the Ptest and Ptrain sets is a matrix of size 960 x 720. 
The variable "names" contains the names of each of the 10 species, and the variable "Prgb" contains rgb images of the leaves that can be used as a reference.

The goal for this project is to build a program in MATLAB that can be trained and then can accurately identify a leaf’s species just based on its image. 
We want to use the Ptrain set of images to train our program and then the Ptest set of images to test it. At the end, the program should achieve an accuracy of about 80%.

## How to run

I recommend running this file in MATLAB Desktop or MATLAB Online, but any IDE that is configured to run MATLAB files would also work. The dataset that I linked above needs to be downloaded
and saved in the same folder as the Leaf_Classification.m file, and when running the program for the first time, it is important to remember to uncomment the line at the top of the file 
(load('LeafData.mat', 'Ptrain', 'Ptest', 'names', 'Prgb');) so that the dataset can be loaded properly. Since this is a large dataset, there is no need to load the dataset everytime you 
run the program, so after running the program for the first time, this line can be commented out. There are also some lines in the middle of the file that can build a convergence plot if 
needed, but these lines are not necessary for the program to run as they were merely used as a test, so I have commented them out.
