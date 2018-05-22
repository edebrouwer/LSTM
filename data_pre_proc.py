#We need to preprocess the input so that later computations are made easier. THIS IS THE ACTUAL PREPROC FUNCTION
import pandas as pd
import multiprocessing as mp

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

num_workers=mp.cpu_count()-2
print(num_workers)

def fun_custom(hadm_num,death_tag):
    a=lab_short.loc[lab_short["HADM_ID"]==hadm_num].pivot(index="LABEL_CODE",columns="TIME_STAMP",values="VALUENUM")
    a["HADM_NUM"]=hadm_num
    a["DEATHTAG"]=death_tag
    a["LABEL_CODE"]=a.index
    return(a)

for idx in range(int(length/num_workers)):#range(length):
    print(idx/int(length/num_workers))
    idxs=list(range(num_workers*idx,num_workers*(idx+1))) # Attention last batch !
    hadm_nums=[death_tags.iloc[idh]["HADM_ID"] for idh in idxs ]
    dtags=[int(death_tags.iloc[idt]["DEATHTAG"]) for idt in idxs]

    pool=mp.Pool(processes=num_workers)
    results=[pool.apply(fun_custom,args=(hadm_num,death_tag,)) for hadm_num,death_tag in zip(hadm_nums,dtags)]

    #a=lab_short.loc[lab_short["HADM_ID"]==hadm_num].pivot(index="LABEL_CODE",columns="TIME_STAMP",values="VALUENUM")
    #a["HADM_NUM"]=hadm_num
    #a["DEATHTAG"]=death_tag
    #a["LABEL_CODE"]=a.index
    results.insert(0,df_final)
    df_final=pd.concat(results)
    pool.close()

    df_final.reset_index()

df_final.to_csv("df_final.csv")
