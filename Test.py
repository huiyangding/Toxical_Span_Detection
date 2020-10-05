import glob

pathToScan = '/shared/0/datasets/reddit/perspective/*.tsv.gz'
for filePath in glob.glob(pathToScan):
    print(filePath)
