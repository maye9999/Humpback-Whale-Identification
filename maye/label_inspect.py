import numpy as np
from collections import defaultdict

labels = defaultdict(int)
total_num = 0

with open('train.csv') as fin:
    for line in fin:
        if line.startswith('Image'):
            continue
        l = line.strip().split(',')
        labels[l[1]] += 1
        total_num += 1

labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)

print(total_num)
print(labels[0][1] / total_num)
print(labels[:50])
print(labels[1000])
print(labels[3000])
from matplotlib import pyplot

# pyplot.plot(np.arange(len(labels)-1), [labels[i][1] for i in range(1, len(labels))])
#
# pyplot.show()
