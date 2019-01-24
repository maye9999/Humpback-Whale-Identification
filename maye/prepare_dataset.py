from pandas import read_csv
from tqdm import tqdm
from random import shuffle
from random import random
all = dict([(p, w) for _, p, w in read_csv('../data-raw/train.csv').to_records()])

print(len(all), list(all.items())[:5])

y2xs = {}

for x, y in all.items():
    if y not in y2xs:
        y2xs[y] = [x]
    else:
        y2xs[y].append(x)

train = []
val = []

for y, xs in y2xs.items():
    y2xs[y] = shuffle(xs)
    if len(xs) == 1:
        pass
    else:
        cnt = 0
        for x in xs:
            if random() >= 0.8:
                cnt += 1
                if cnt < len(xs):
                    val.append(x)
                else:
                    train.append(x)
            else:
                train.append(x)

print(len(train), len(val), len(train) / float(len(val)))

shuffle(train)
shuffle(val)

tt = set()

# for i in train:
#     tt.add(i)
for i in val:
    tt.add(i)

print(len(tt))
# with open('../train-new2.csv', 'w') as fout:
#     fout.write("Image,Id\n")
#     for x in train:
#         fout.write("%s,%s\n" % (x, all[x]))
#
# with open('../val2.csv', 'w') as fout:
#     fout.write("Image,Id\n")
#     for x in val:
#         fout.write("%s,%s\n" % (x, all[x]))
