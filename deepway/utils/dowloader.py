import os
import json
import argparse
import pandas as pd
import urllib.request
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-p1", "--path1", help="provide image folder name")
parser.add_argument("-p2", "--path2", help="provide mask folder name")
parser.add_argument("-c", "--csv", help="path to csv file")
args = parser.parse_args()
# os.mkdir(args.path1)
# os.mkdir(args.path2)

data = pd.read_csv(args.csv)
print(len(data))
for i in tqdm(range(len(data))):
    r = data.iloc[i,2]
    urllib.request.urlretrieve(r, args.path1+"/"+str(i)+'.jpg')
    p = data.iloc[i, 3]
    if isinstance(p, str):
        try:
            r = json.loads(p)["objects"][0]["instanceURI"]
            urllib.request.urlretrieve(r, args.path2+"/"+str(i)+'.png')
        except:
            os.remove("img/"+str(i)+".jpg")
print("completed")


