import numpy as np
import pylab
import scipy.stats as stats
import urllib2
import sys

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

# read the data
data = urllib2.urlopen(target_url)

# arrange data into lists
xList = []
labels = []
for line in data:
    row = line.strip().split(",")
    xList.append(row)
num_rows = len(xList)
num_cols = len(xList[1])
print(num_rows)
print(num_cols)
type = [0]*3
colCounts = []

# generate summary statistics for column 3
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))

# plotting the quantile-quantile of attribute number 4, helps to
# detect outliers
stats.probplot(colData, dist="norm", plot=pylab)
pylab.savefig("plots/quantile-quantile-attribute-4.png")
pylab.show()
