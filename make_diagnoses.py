import pandas as pd

diagnoses_codes = pd.read_csv('dx_mapping.csv', usecols=['SNOMED CT Code', 'Dx'])

diagnoses_dict = {}

for diagnosis in diagnoses_codes.values:
    #print(diagnosis)
    diagnoses_dict[diagnosis[1]] = diagnosis[0]

print(diagnoses_dict)

df = pd.DataFrame.from_dict(diagnoses_dict, orient="index")
print(df)

df.to_csv('diagnoses.csv')
