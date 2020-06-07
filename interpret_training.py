training_log = '/Users/hadi/Downloads/Log.txt'

file = open(training_log, 'r')

for l in file:
    if 'loss' in l:
        line = l.split(' ')
        print(line[3])