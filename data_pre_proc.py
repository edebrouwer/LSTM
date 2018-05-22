#We need to preprocess the input so that later computations are made easier. THIS IS THE ACTUAL PREPROC FUNCTION
import pandas as pd

input_dim=29
csv_file_serie="lab_events_short.csv"
csv_file_tag="death_tags.csv"
file_path="~/Data/MIMIC/"
lab_short=pd.read_csv(file_path+csv_file_serie)
lab_short=lab_short.drop_duplicates(["LABEL_CODE","TIME_STAMP","HADM_ID"])
death_tags=pd.read_csv(file_path+csv_file_tag)
length=len(death_tags.index)
input_dim=input_dim

col_list=([str(i) for i in range(101)])
col_list.insert(0,"HADM_ID")
col_list.append("DEATHTAG")

df_final=pd.DataFrame(columns=col_list)
for idx in range(length):
    print(idx/length)
    hadm_num=death_tags.iloc[idx]["HADM_ID"]
    death_tag=int(death_tags.iloc[idx]["DEATHTAG"])

    a=lab_short.loc[lab_short["HADM_ID"]==hadm_num].pivot(index="LABEL_CODE",columns="TIME_STAMP",values="VALUENUM")
    a["HADM_NUM"]=hadm_num
    a["DEATHTAG"]=death_tag
    a["LABEL_CODE"]=a.index
    df_final=pd.concat([df_final,a])
    #df_final.reset_index()

df_final.to_csv("df_final.csv")
