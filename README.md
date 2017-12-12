This project uses matlab and python 2.7.

Original dataset (acm_output.txt) located at: https://aminer.org/citation  (V6)

Data for matlab code found at:
https://drive.google.com/open?id=1OzSAG30jifiRd536JfusQrYVv-e4lX48

The libraries being used in python are as follows:
1. pandas
2. numpy
3. gensim
4. logging
5. sklearn
6. scipy
7. matplotlib

How to run the code:
1) Run the parser.py
2) It will create the output.csv file
3) Run plots.py
4) Run knn_.m (KNN), Lin_Reg.m (Linear Regression), NL_reg.m (non-Linear Regression), regression_tree.m (Decision Tree), SVM_linear.m (SVM)
5) Run MLP.py which is the Neural network regressor
6) Run citation2class.py (creates labels for classification), classification.py (Classification)

Note that we ran the KNN and classification files (knn_.m, classification.py) on a server due to the lack of memory on personal machines.

File Desciption
1) acm_output.txt is the original file (dataset)
2) output.csv is the result of parsing (pre_precessed file)
3) outputt.mat is the converted output of the csv verison of the file
4) test_ind.mat is the set of indices for training and test
5) labels.csv is the labels for classification
