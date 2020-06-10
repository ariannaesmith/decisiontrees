Created by: Arianna Smith
CSE 6375

Homework 1
A program to implement the decision tree algorithm with both information gain and variance impurity, reduced-error post pruning, depth-based pruning, and scikit-learn random forests.

To run:
Have all datasets and .py files in the same folder. From command line, navigate to the folder and then run as arguments:
0- drivercode.py
1- “clause size”
2- “sample number”
3- RF *or* “impurity type”
4- “prune method”

“Clause size” options are: c300, c500, c1000, c1500, c1800
“Sample number” options are: d00, d1000, d5000
“Impurity type” options are: entropy, variance
“Prune method” options are: error, depth, none


Written with: Python 3.7
Written in: Spyder 3.3.6
Written on: macOS Catalina Version 10.15.1


Files: 
decisiontree.py
drivercode.py
heuristics.py
pruning.py
randomforests.py
testfunction.py

test_c300_d100.csv
test_c300_d1000.csv
test_c300_d5000.csv
test_c500_d100.csv
test_c500_d1000.csv
test_c500_d5000.csv
test_c1000_d100.csv
test_c1000_d1000.csv
test_c1000_d5000.csv
test_c1500_d100.csv
test_c1500_d1000.csv
test_c1500_d5000.csv
test_c1800_d100.csv
test_c1800_d1000.csv
test_c1800_d5000.csv
train_c300_d100.csv
train_c300_d1000.csv
train_c300_d5000.csv
train_c500_d100.csv
train_c500_d1000.csv
train_c500_d5000.csv
train_c1000_d100.csv
train_c1000_d1000.csv
train_c1000_d5000.csv
train_c1500_d100.csv
train_c1500_d1000.csv
train_c1500_d5000.csv
train_c1800_d100.csv
train_c1800_d1000.csv
train_c1800_d5000.csv
valid_c300_d100.csv
valid_c300_d1000.csv
valid_c300_d5000.csv
valid_c500_d100.csv
valid_c500_d1000.csv
valid_c500_d5000.csv
valid_c1000_d100.csv
valid_c1000_d1000.csv
valid_c1000_d5000.csv
valid_c1500_d100.csv
valid_c1500_d1000.csv
valid_c1500_d5000.csv
valid_c1800_d100.csv
valid_c1800_d1000.csv
valid_c1800_d5000.csv