import openpyxl, pandas as pd, numpy as np, json, sys
U='./datos/'  # carpeta con los 3 archivos de la empresa
FILES={'EMP':U+'RECAUDO_EMPRESARIAL_2016_2026_UNIFICADO_1.xlsx','IND':U+'RECAUDO_INDIVIDUAL_2016_2026_UNIFICADO_1.xlsx'}
rows=[]; checks=[]; sheetinfo=[]
for lib,f in FILES.items():
    wb=openpyxl.load_workbook(f,read_only=True,data_only=True)
    for s in wb.sheetnames:
        y=int(s.split()[-1]); ws=wb[s]
        data=list(ws.iter_rows(values_only=True))
        hdr=data[0]; body=data[1:]
        n_tot=0; n_empty=0; n_neg=0; pmis=0; nrows=0
        for i,r in enumerate(body, start=2):
            if r is None or all(v is None for v in r): n_empty+=1; continue
            if r[0] and str(r[0]).strip().upper().startswith('TOTAL'): n_tot+=1; continue
            nrows+=1
            vals=[(v if isinstance(v,(int,float)) else 0) or 0 for v in r[3:15]]
            if any(v<0 for v in vals): n_neg+=1
            if r[15] is not None and isinstance(r[15],(int,float)) and abs(sum(vals)-r[15])>1: pmis+=1
            for m,v in enumerate(vals,1):
                rows.append(dict(libro=lib,anio=y,mes=m,asesor=str(r[0]).strip() if r[0] else '',contrato=r[1],nombre=str(r[2]).strip() if r[2] else '',fila=i,valor=float(v)))
        sheetinfo.append(dict(libro=lib,anio=y,filas=nrows,total_rows=n_tot,vacias=n_empty,negativos=n_neg,p_mismatch=pmis))
df=pd.DataFrame(rows)
df.to_parquet('long.parquet') if False else df.to_pickle('long.pkl')
si=pd.DataFrame(sheetinfo); print(si.to_string())
d=df[df.valor!=0]
print(df.groupby(['libro','anio']).valor.sum().unstack(0)/1e6)
print('contratos', df.groupby('libro').contrato.nunique(), 'union', df.contrato.nunique(), 'ambos', len(set(df[df.libro=='EMP'].contrato)&set(df[df.libro=='IND'].contrato)))
print('filas', si.groupby('libro').filas.sum())
