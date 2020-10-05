import pandas as pd
import numpy as np
import glob
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def label_by_PerspScore (row,
                         offensiveBound: float,
                         nonoffensiveBound: float
                       ):
    """
    This function labels the offensive/non-offensive comments based on the Perspective score
    """
    try:
        if row["Perpective_Score"] >= offensiveBound:
            return 1
        elif row["Perpective_Score"] <= nonoffensiveBound:
            return 0
        else:
            return 'NA'
    except Exception as e:
        print(e)

def make_Explanations (classifier,
                      explainer,
                      row,
                      num_features):
    exp = explainer.explain_instance(row["body"], classifier.predict_proba, num_features = num_features)
    expList = exp.as_list(label = 1)
    return expList

def extracting_Spans (row):
    curRow = re.findall(r"\(.*?\)",row['Lime'])
    curList = [i.split(",")[0][2 : -1] for i in curRow]
    return curList

def printing_Results(y_test, pred_rf):
    accuracy = accuracy_score(y_test, pred_rf)
    precision = precision_score(y_test, pred_rf, average='weighted')
    recall = recall_score(y_test, pred_rf, average='weighted')
    f1 = f1_score(y_test, pred_rf, average='weighted')
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

# def bin_by_PerspScore (pathToScan: str,
#                        fileFormat: str,
#                        pathToSave: str,
#                        lowerLimit: float,
#                        upperLimit: float,
#                        step: float,
#                        sampleSize: int
#                        ):
#     for filePath in glob.glob(pathToScan + '/*.' + fileFormat):
#         fileName = filePath[filePath.find('RC_'): filePath.find('.' + fileFormat)]
#         try:
#             tmpDF = pd.read_csv(filePath)
#             tmpDF.columns = ['id', 'Perpective_Score']
#             bins = np.arange(lowerLimit, upperLimit + step, step)
#             tmpDFSampled = tmpDF.drop_duplicates().sample(n=sampleSize, random_state=32, replace=False)
#             tmpDFSampled['Binned'] = pd.cut(tmpDFSampled['Perpective_Score'], bins)
#             tmpDFSampled['BinnedCode'] = tmpDFSampled['Binned'].cat.codes
#             mappingCode = dict(enumerate(tmpDFSampled['Binned'].cat.categories))



def checkToxicalWords (sampledText, index):
    toxicalWords = []
    sepCharList = [list(group) for group in mit.consecutive_groups(literal_eval(Toxical.iloc[index][0]))]
    charList = []
    for word in sepCharList:
        charList.append(list(sampledText[int(i)] for i in word)) #Extract character from there
    spanList = [None] * len(charList)
    for i in range(len(charList)):
        spanList[i] = str(''.join(charList[i]))
    return spanList

