import pandas as pd

datapath = 'output.csv'
'''
load data
'''
print '\nloading data...'

Xy = pd.read_csv(datapath, header = None, skiprows = 1, usecols = range(0,69)) # y is column at index = 2, X is the remaining columns

# seperate output (y) matrix from full data matrix
y = Xy.iloc[:, 0:3] # keep orignal index to sort ynew into y's original order
del y[1]
ynew = pd.DataFrame(index=[i for i in xrange(len(y))], columns=['class'])

'''
binary labelling: citation vs no citation
'''
print '\nconverting citations to classes...'

index = 0
counts = [0 for i in xrange(2)]
while index < len(y):
	if y.values[index][1] == 0:
		counts[0] += 1
		ynew.set_value(index, 'class', 0)
	else:
		counts[1] += 1
		ynew.set_value(index, 'class', 1)

	index += 1

#count and print number of data in each class
for index, count in enumerate(counts):
	print str(index) + ': ', count

'''
save label matrix as .csv
'''
print '\nsaving labels...\n'

ynew.to_csv('labels.csv')
