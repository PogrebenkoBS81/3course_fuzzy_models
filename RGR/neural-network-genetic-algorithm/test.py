import pandas as pd
import pickle
from network import Network

classifier = None

with open("networks/classifier_2021-01-08_acc0.8846153846153846.pkl", 'rb') as fid:
    classifier = pickle.load(fid)

df = pd.read_csv('Dataset_spine_test.csv')
# Drop dummy column.
df = df.drop(['Unnamed: 13'], axis=1)
# Rename columns according to: https://towardsdatascience.com/an-exploratory-data-analysis-on-lower-back-pain-6283d0b0123.
df.rename(columns={
        "Col1" : "pelvic_incidence",
        "Col2" : "pelvic_tilt",
        "Col3" : "lumbar_lordosis_angle",	
        "Col4" : "sacral_slope",
        "Col5" : "pelvic_radius",	
        "Col6" : "degree_spondylolisthesis",
        "Col7" : "pelvic_slope",
        "Col8" : "Direct_tilt",	
        "Col9" : "thoracic_slope",	
        "Col10" : "cervical_tilt",	
        "Col11" : "sacrum_angle",	
        "Col12" : "scoliosis_slope",	
    }
)

x_test = df.drop(['Class_att'], axis=1)
print(x_test)

y_pred = classifier.clf.predict(x_test)
print(y_pred)