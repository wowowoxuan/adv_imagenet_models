import numpy as np
import csv
import pandas as pd

a = np.loadtxt('ILSVRC2012_validation_ground_truth.txt', dtype = int)
print(len(a))
print(a)

#df1=pd.read_csv('./valid_gt.csv')
#with open('./valid_gt.csv', mode='r') as infile:
# with open('./valid_gt.csv','r') as data: 
#    for line in csv.DictReader(data): 
#         print(line) 
# a_csv_file = open("./valid_gt.csv", "r")
# dict_reader = csv.DictReader(a_csv_file)

# ordered_dict_from_csv = list(dict_reader)[0]
# dict_from_csv = dict(ordered_dict_from_csv)
# print(dict_from_csv[('name', 'ILSVRC2012_val_00049997')])

# reader = csv.reader(open('./valid_gt.csv'))

# result = {}
# for row in reader:
#     print(row)
#     break
#     # key = row.pop('Date')
#     # if key in result:
#     #     # implement your duplicate row handling here
#     #     pass
#     # result[key] = row
# print(result)
result = {}
with open('./valid_gt.csv') as f:
       for line in f:
            fields = line.split(',')

            (key,  value) = fields
            # print(key)
            # print(value)
            result[key] = value
print(int(result['ILSVRC2012_val_00049997']))