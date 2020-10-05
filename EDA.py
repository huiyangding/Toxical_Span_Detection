"""
Project: This is for EDA of the Reddit datasets
Time Created: 9/24/2020
Author: Huiyang Ding
Mentor: David Jurgens
Reference: SemEval https://competitions.codalab.org/competitions/25623
"""

## Libraries
import pandas as pd
import numpy as np
import lzma
import functions
import glob
import re
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import dask.dataframe as dd

## Scan the directory
##Parameters

# pathToScan = '/shared/0/datasets/reddit/perspective/*.tsv.gz'
# for filePath in glob.glob(pathToScan):
#     fileName = filePath[filePath.find('RC_') : filePath.find('.tsv.gz')]
#     RC_2018_01 = pd.read_csv('/shared/0/datasets/reddit/perspective/RC_2018-01.tsv.gz',
#                              header = None,
#                              compression = 'gzip',
#                              sep = '\t')
#     print(fileName)






# path = "/home/huiyangd/toxicSpans/data"
# fileName1 = "2018_01_n_1000_32.csv"
# fileName2 = "2018_01_p_1000_32.csv"
#
# RC_2018_01_n_1000_32 = pd.read_csv(path + '/' + fileName1)
# RC_2018_01_p_1000_32 = pd.read_csv(path + '/' + fileName2)
# frames = [RC_2018_01_n_1000_32, RC_2018_01_p_1000_32]
# RC_2018_01_combined_1000 = pd.concat(frames)
#
# print(list(RC_2018_01_combined_1000.columns))
#
# class_names = list(RC_2018_01_combined_1000.toxicity.unique())
# limeExplainer = LimeTextExplainer(class_names = class_names)
# idx = 1083
# print(class_names)
# print(RC_2018_01_combined_1000.iloc[idx].loc['body'])
# print(RC_2018_01_combined_1000.iloc[idx].loc['Perpective_Score'])
# exp = limeExplainer.explain_instance(RC_2018_01_combined_1000.iloc[idx].loc['body'], RC_2018_01_combined_1000.iloc[idx].loc['Perpective_Score'], num_features = 10)
# print('Document id: %d' % idx)
# print('Probability(christian) =', RC_2018_01_combined_1000.Perpective_Score[idx])
# print('True class: %s' % class_names[RC_2018_01_combined_1000.toxicity[idx]])






# RC_2018_01 = pd.read_csv('/shared/0/datasets/reddit/perspective/RC_2018-01.tsv.gz',
#                          header = None,
#                          compression = 'gzip',
#                          sep = '\t')
# RC_2018_01.columns = ['id', 'Perpective_Score']
# #
# ##Parameters
# # sampleSize = 1000
# upperLimit = 1.0
# lowerLimit = 0.0
# step = 0.05
# bins = np.arange(lowerLimit, upperLimit + step, step)
#
# ## Adding bins and encoding them into categories
# RC_2018_01_sample = RC_2018_01.drop_duplicates()
# RC_2018_01_sample['Binned'] = pd.cut(RC_2018_01_sample['Perpective_Score'], bins)
# RC_2018_01_sample['BinnedCode'] = RC_2018_01_sample['Binned'].cat.codes
# mappingCode = dict(enumerate(RC_2018_01_sample['Binned'].cat.categories))
# print(mappingCode)
# RC_2018_01_sample.head(20)
# print(RC_2018_01_sample['BinnedCode'].value_counts(dropna = False))


#     curFile = pd.read_csv(filePath,
#                          header = None,
#                          compression = 'gzip',
#                          sep = '\t')
#     curFile.columns = ['id', 'Perpective_Score']
#     curFile = curFile[
#         (curFile['Perpective_Score'] >= offensiveCutoff) | (curFile['Perpective_Score'] <= nonOffensiveCutoff)]
#     curFile_no_dup = curFile.drop_duplicates()
#     curFile_no_dup['toxicity'] = curFile_no_dup.apply(lambda row: functions.label_by_PerspScore(row,
#                                                                                                 offensiveCutoff,
#                                                                                                 nonOffensiveCutoff),
#                                                       axis=1)

RC_2018_01 = pd.read_csv('/shared/0/datasets/reddit/perspective/RC_2018-01.tsv.gz',
                         header = None,
                         compression = 'gzip',
                         sep = '\t')
RC_2018_01.columns = ['id', 'Perpective_Score']
offensiveCutoff = 0.70
nonOffensiveCutoff = 0.30
sampleSize = 100000
RC_2018_01 = RC_2018_01.drop_duplicates()
RC_2018_01 = RC_2018_01[(RC_2018_01['Perpective_Score'] >= offensiveCutoff) | (RC_2018_01['Perpective_Score'] <= nonOffensiveCutoff)]
RC_2018_01['toxicity'] = RC_2018_01.apply(lambda row: functions.label_by_PerspScore(row,
                                                                                    offensiveCutoff,
                                                                                    nonOffensiveCutoff),
                                          axis = 1)

RC_2018_01["body"] = ""
original_Dataset = '/shared/2/datasets/reddit-dump-all/RC/RC_2018-01.xz'
# stop_limit = 10
# n = 0
with lzma.open(original_Dataset, mode = "rt") as file:
    for line in file:
        curId = line[line.find("\"id\":") + 6 : line.find(",\"is_submitter\":") - 1]
        # if (n == stop_limit):
        #     break
        try:
            RC_2018_01.loc[RC_2018_01['id'] == curId, 'body'] \
                = line[line.find("\"body\":") + 8: line.find(",\"can_gild\":") - 1]
            # n += 1
        except:
            print("%s. not found" % curId)
            continue


RC_2018_01.to_csv("/shared/0/projects/toxic-spans/data/RC_2018_01_Text_Combined.csv")


# RC_2018_01.columns = ['id', 'Perpective_Score']
#
# # sampleSize = 1000
# upperLimit = 1.0

# lowerLimit = 0.0
# step = 0.05
# bins = np.arange(lowerLimit, upperLimit + step, step)
#
# ## Adding bins and encoding them into categories
# RC_2018_01['Binned'] = pd.cut(RC_2018_01['Perpective_Score'], bins)
# RC_2018_01['BinnedCode'] = RC_2018_01['Binned'].cat.codes
# mappingCode = dict(enumerate(RC_2018_01['Binned'].cat.categories))
# print(mappingCode)

