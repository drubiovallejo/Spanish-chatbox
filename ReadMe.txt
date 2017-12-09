# Spanish chatbox v.1
# Author: David Rubio Vallejo
# Email: drubio@brandeis.edu
# Github : https://github.com/drubiovallejo/Spanish-chatbox

This folder contains the code to run a Spanish chatbot created as a final project for the class "Ling131: NLP with Python" during my first semester in the CL MA at Brandeis University. All the code in it was written by the author.

The chatbot was trained using selected files from the CORLEC oral corpus of contemporary Spanish (http://www.lllf.uam.es/ESP/Info%20Corlec.html), which are also included here. This project contains 3 python files:

- Main.py: Contains the client code and initializes the chatbot. The user can choose from here either of the 3 available algorithms for the chatbot to select its output: a random sentence selector, a selection function based on the Jaccard coefficient, and a selection function based on the TF-IDF measure.

- Selection_methods.py: Contains the implementation of the selection algorithms described above.

- Preprocessing.py: Contains the code to process the corpus files, extract the desired information and train the model. 

From the very first interactions with the chatbot, it becomes clear that its output possibilities are very limited (partly, because of the small size of the training data). It is my hope to be able to continue developing this project as I learn more about NLP techniques.