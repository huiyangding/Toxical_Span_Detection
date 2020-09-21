"""
Project: Toxic Span Detection - Blablab Lab
Time Created: 9/19/2020
Author: Huiyang Ding
Mentor: David Jurgens
Reference: https://competitions.codalab.org/competitions/25623
"""

## Libraries
import pandas as pd
import numpy as np
import gzip
import nltk
import lzma
import functions
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

#2018 Reddit Perspective.ai annotated data is stored at: /shared/0/datasets/reddit/perspective/
RC_2018_01 = pd.read_csv('/shared/0/datasets/reddit/perspective/RC_2018-01.tsv.gz',
                         header = None,
                         compression = 'gzip',
                         sep = '\t')
RC_2018_01.columns = ['id', 'Perpective_Score']

##Parameters
offensiveCutoff = 0.70
nonOffensiveCutoff = 0.30
sampleSize = 1000

RC_2018_01 = RC_2018_01[(RC_2018_01['Perpective_Score'] >= offensiveCutoff) | (RC_2018_01['Perpective_Score'] <= nonOffensiveCutoff)]
RC_2018_01_sample = RC_2018_01.drop_duplicates().sample(n = sampleSize, random_state = 32, replace = False)
RC_2018_01_sample['toxicity'] = RC_2018_01_sample.apply(lambda row: functions.label_by_PerspScore(row,
                                                                                                  offensiveCutoff,
                                                                                                  nonOffensiveCutoff),
                                                        axis = 1)
# print(RC_2018_01_sample.head())
"""
print(RC_2018_01.drop_duplicates().shape)
(67619129, 2)

print(RC_2018_01_sample.head())
               id  Perpective_Score  toxicity
28275370  dsnq5y9          0.096382         0
4708167   ds4imve          0.106988         0
64517151  dth6hcm          0.070802         0
60017570  dtdjcfo          0.920156         1
62242270  dtfcnf8          0.073084         0
"""

canIdList = RC_2018_01_sample["id"].to_list()
RC_2018_01_sample["body"] = "" # The place to hold the text body
early_Stopper = 0 # Stop the traverse whenever we get the right sample size
with lzma.open('/shared/2/datasets/reddit-dump-all/RC/RC_2018-01.xz', mode='rt') as file:
    for line in file:
        curId = line[line.find("\"id\":") + 5 : line.find(",\"is_submitter\":")]
        if (curId in canIdList):
            RC_2018_01_sample.at[RC_2018_01_sample[RC_2018_01_sample["id"] == curId].index[0], "body"] \
                = line[line.find("\"body\":") + 7 : line.find(",\"can_gild\":")]
            early_Stopper += 1
        # if early_Stopper == sampleSize:
        if early_Stopper == 1:
            break

class_names = list(RC_2018_01_sample.toxicity.unique())
explainer = LimeTextExplainer(class_names = class_names)
idx = 1877
exp = explainer.explain_instance(RC_2018_01_sample[idx], c.predict_proba, num_features=6, labels=[4, 8])
print('Document id: %d' % idx)
print('Predicted class =', class_names[logreg.predict(test_vectors[idx]).reshape(1,-1)[0,0]])
print('True class: %s' % class_names[y_test[idx]])

"""
print(count)
91558594

Sample:
{"author":"wardog45",
"author_flair_css_class":"33",
"author_flair_text":"",
"body":"Ouch",
"can_gild":true,
"controversiality":0,
"created_utc":1514765865,
"distinguished":null,
"edited":false,
"gilded":0,
"id":"ds0ny2p",   ## This is the commend id
"is_submitter":false,
"link_id":"t3_7nas1f",
"parent_id":"t3_7nas1f",
"permalink":"/r/49ers/comments/7nas1f/game_thread_san_francisco_49ers_at_los_angeles/ds0ny2p/",
"retrieved_on":1517436335,
"score":1,
"stickied":false,
"subreddit":"49ers",
"subreddit_id":"t5_2rebv",
"subreddit_type":"public"}
"""



