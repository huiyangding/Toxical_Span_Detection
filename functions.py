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

