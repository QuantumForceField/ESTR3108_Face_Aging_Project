#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


def saveEpochHistory(filename, adLossList, ganLossList, epochsBetweenEachHistory):
    # epochsBetweenEachHistory means how many epochs were passed between each epochs history point
    
    
    # create new temp file
    with open(filename + '_temp.txt', 'w') as f:
        f.write('ADLoss\tGANLoss\tEpochsBetweenEachHistory=' + str(epochsBetweenEachHistory) + '\n')
        for i in range(len(adLossList)):
            f.write(str(adLossList[i]) + '\t' + str(ganLossList[i]) + '\n');
    
    if os.path.isfile(filename):
        os.remove(filename)
    
    # rename temp file
    os.rename(filename + '_temp.txt', filename)


# In[ ]:


# Give a file containing epoch histories
# Return epochsBetweenEachHistory, adLossList, ganLossList

def loadEpochHistory(filename):
    
    if os.path.isfile(filename):
        epochsBetweenEachHistory = 0
        adLossHis = []
        ganLossHis = []
        with open(filename, 'r') as f:
            counter = 0
            for line in f:
                if counter == 0:
                    epochsBetweenEachHistory = int(line.split('=')[-1])
                else:
                    result = line.split('\t')
                    adLossHis.append(float(result[0]))
                    ganLossHis.append(float(result[1]))
                counter = counter + 1
        return epochsBetweenEachHistory, adLossHis, ganLossHis
    else:
        print('Cannot load epoch history file \"' + filename + '\", return 0, None, None')
        return 0, None, None
            

