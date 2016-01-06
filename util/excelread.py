import xlrd
from collections import defaultdict
import re
import random

input = 'C:\\Users\\xiaohucheng\\Desktop\\公众平台1个月log数据\\1周.xls'
output = 'C:\\Users\\xiaohucheng\\Desktop\\公众平台1个月log数据\\output.txt'

fout = open(output, 'w')
data = xlrd.open_workbook(input)
sheets_len = len(data.sheets())
for i in range(0, sheets_len):
    table = data.sheets()[i]
    nrows = table.nrows
    for j in range(0, nrows):
        fout.write(table.cell(j, 0).value + '\t' + table.cell(j, 1).value + '\t' + table.cell(j, 2).value.replace('\t', ''))
fout.close()
