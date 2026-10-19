import pandas as pd, numpy as np, json, warnings
from pyxlsb import open_workbook
warnings.filterwarnings('ignore')
U='./datos/'  # carpeta con los 3 archivos de la empresa
df=pd.read_pickle('long.pkl')
df['FV']=df.nombre.str.upper().str.contains('DAVIVIENDA')
idx=pd.period_range('2016-01','2026-09',freq='M')
def ser(mask):
    s=df[mask].groupby(['anio','mes']).valor.sum()
    s.index=[pd.Period(f'{a}-{m:02d}','M') for a,m in s.index]
    return s.reindex(idx).fillna(0)/1e6
S={'EMP':ser(df.libro=='EMP'),'IND':ser(df.libro=='IND')}
S['TOT']=S['EMP']+S['IND']
S['EMP_sFV']=ser((df.libro=='EMP')&~df.FV); S['IND_sFV']=ser((df.libro=='IND')&~df.FV)
S['TOT_sFV']=S['EMP_sFV']+S['IND_sFV']
S['FV']=S['TOT']-S['TOT_sFV']
out={'meses':[str(p) for p in idx],'series':{k:[round(v,3) for v in s.values] for k,s in S.items()}}

# ---- Caracterización
def jansep(s,y): return s[[p for p in s.index if p.year==y and p.month<=9]].sum()
car={}
for k in ['EMP','IND','TOT','EMP_sFV','IND_sFV','TOT_sFV']:
    car[k]={'anual':{y:round(S[k][[p for p in idx if p.year==y]].sum(),1) for y in range(2016,2027)},
            'js25':round(jansep(S[k],2025),1),'js26':round(jansep(S[k],2026),1)}
    car[k]['crec_js']=round((car[k]['js26']/car[k]['js25']-1)*100,1)
    a16=car[k]['anual'][2016]; a25=car[k]['anual'][2025]
    car[k]['cagr']=round(((a25/a16)**(1/9)-1)*100,2)
for k in car: print(k,car[k]['js25'],car[k]['js26'],car[k]['crec_js'],car[k]['cagr'])
# peso por mes 2016-2025
peso={}
for k in ['EMP','IND','TOT','TOT_sFV','EMP_sFV']:
    s=S[k][[p for p in idx if p.year<=2025]]
    tab=pd.DataFrame({'y':[p.year for p in s.index],'m':[p.month for p in s.index],'v':s.values})
    tab['sh']=tab.v/tab.groupby('y').v.transform('sum')
    peso[k]=[round(x*100,2) for x in tab.groupby('m').sh.mean().values]
car['peso_mes']=peso
print('peso julio',{k:v[6] for k,v in peso.items()})
# FV table
fvt=[]
for y in range(2016,2027):
    yy=[p for p in idx if p.year==y]
    fv=S['FV'][yy]; 
    fvt.append({'anio':y,'emp':round(S['EMP'][yy].sum(),1),'fv':round(fv.sum(),1),'mes_max':int(fv.values.argmax()+1) if fv.sum()>0 else None,'fv_max':round(fv.max(),1),'pct_mes':round(100*fv.max()/S['TOT'][yy].values[fv.values.argmax()],1) if fv.sum()>0 else 0})
car['fv']=fvt
# contratos: variación 2026 vs 2025 ene-sep
d=df[df.mes<=9]
g=d[d.anio.isin([2025,2026])].groupby(['libro','contrato','anio']).valor.sum().unstack(fill_value=0)/1e6
names=df.sort_values('anio').groupby(['libro','contrato']).nombre.last()
g['delta']=g[2026]-g[2025]; g['nombre']=names.reindex(g.index).values
contr={}
for lib in ['EMP','IND']:
    gg=g.loc[lib].sort_values('delta')
    contr[lib]={'neg':[{'c':int(c),'n':r.nombre,'v25':round(r[2025],1),'v26':round(r[2026],1),'d':round(r.delta,1)} for c,r in gg.head(8).iterrows()],
                'pos':[{'c':int(c),'n':r.nombre,'v25':round(r[2025],1),'v26':round(r[2026],1),'d':round(r.delta,1)} for c,r in gg.tail(8)[::-1].iterrows()],
                'neto':round(gg.delta.sum(),1),'n25':int((gg[2025]>0).sum()),'n26':int((gg[2026]>0).sum())}
    print(lib,contr[lib]['neto']); print(gg.head(5)[[2025,2026,'delta','nombre']]); print(gg.tail(5)[[2025,2026,'delta','nombre']])
car['contratos']=contr
# top contratos por mes 2026 (pico)
top=[]
d26=df[(df.anio==2026)&(df.mes<=9)]
for m in range(1,10):
    t=d26[d26.mes==m].groupby(['libro','contrato','nombre']).valor.sum().sort_values(ascending=False).head(3)/1e6
    tot=d26[d26.mes==m].valor.sum()/1e6
    top.append({'mes':m,'total':round(tot,1),'top':[{'l':l,'c':int(c),'n':n,'v':round(v,1),'pct':round(100*v/tot,1)} for (l,c,n),v in t.items()]})
car['top_mes']=top
out['car']=car

# ---- outliers via STL
from statsmodels.tsa.seasonal import STL
otl={}
for k in ['TOT','EMP','IND','TOT_sFV']:
    y=np.log(S[k].values)
    r=STL(pd.Series(y,index=idx.to_timestamp()),period=12,robust=True).fit()
    res=r.resid.values; q1,q3=np.percentile(res,[25,75]); iqr=q3-q1
    z=np.where(res>q3,(res-q3)/iqr,np.where(res<q1,(res-q1)/iqr,0))
    fs=max(0,1-np.var(r.resid)/np.var(r.seasonal+r.resid)); ft=max(0,1-np.var(r.resid)/np.var(r.trend+r.resid))
    otl[k]={'atipicos':[{'mes':str(idx[i]),'z':round(float(z[i]),2),'valor':round(float(S[k].values[i]),1)} for i in np.where(np.abs(z)>3)[0]],
            'f_estac':round(fs,3),'f_tend':round(ft,3)}
    print(k,'Fs',round(fs,3),'Ft',round(ft,3),'atip',[(o['mes'],o['z']) for o in otl[k]['atipicos']])
out['stl']=otl

# ---- presupuesto 2026 y cumplimiento
with open_workbook(U+'SEGUIMIENTO_CUMPLIMIENTO_METAS_POR_EJECUTIVO_COMERCIAL_2026_1.xlsb') as wb:
    with wb.get_sheet('2026') as sh: R=[[c.v for c in r] for r in sh.rows()]
    with wb.get_sheet('ejecución 2026') as sh: E=[[c.v for c in r] for r in sh.rows()]
    with wb.get_sheet('presupuesto 2026') as sh: P=[[c.v for c in r] for r in sh.rows()]
bud={}
for r in R:
    if r[0]=='Total Empresarial': bud['EMP']=[float(v) for v in r[1:13]]
    if r[0]=='Total Emp Indv': bud['IND']=[float(v) for v in r[1:13]]
    if r[0]=='Total general': bud['TOT']=[float(v) for v in r[1:13]]
for k in bud: bud[k]=[v/1e6 for v in bud[k]]
print('bud sums',{k:(round(sum(v[:9]),1),round(sum(v),1)) for k,v in bud.items()})
cum={}
for k in ['EMP','IND','TOT']:
    act=[S[k][pd.Period(f'2026-{m:02d}','M')] for m in range(1,10)]
    actsf=[S[k+'_sFV'][pd.Period(f'2026-{m:02d}','M')] for m in range(1,10)]
    b=bud[k]
    cum[k]={'presu':[round(x,2) for x in b],'real':[round(x,2) for x in act],'real_sFV':[round(x,2) for x in actsf],
            'cump_mes':[round(100*a/p,1) for a,p in zip(act,b[:9])],
            'cump_acum':[round(100*np.sum(act[:i+1])/np.sum(b[:i+1]),1) for i in range(9)],
            'cump_acum_sFV':[round(100*np.sum(actsf[:i+1])/np.sum(b[:i+1]),1) for i in range(9)],
            'presu_anual':round(sum(b),1),'presu_js':round(sum(b[:9]),1),'real_js':round(sum(act),1),'real_js_sFV':round(sum(actsf),1)}
    print(k,cum[k]['cump_acum'][-1],cum[k]['cump_acum_sFV'][-1],cum[k]['presu_js'],cum[k]['real_js'])
# reporte por ejecutivo (seudonimizado)
ej=[]
for r in E[1:]:
    if not r[0] or r[13] is None: continue
    ej.append({'nombre':str(r[0]).strip(),'real':[float(v or 0)/1e6 for v in r[1:10]],'acum':float(r[13])/1e6,'meta':float(r[14])/1e6})
# meta mensual por ejecutivo desde 'presupuesto 2026'
pm={str(r[0]).strip().upper():[float(v or 0)/1e6 for v in r[1:13]] for r in P[1:] if r[0]}
ej=sorted(ej,key=lambda x:-x['meta'])
rep={'ejecutivos':[]}
for i,e in enumerate(ej):
    fvadj=998.613 if 'HENAO' in e['nombre'].upper() else 0
    nm=e['nombre'].upper().replace('(CLAUDIA)','').strip()
    rep['ejecutivos'].append({'id':f'E{i+1:02d}','canal':('Canal' if any(w in nm for w in ['VICEPRESIDENCIA','ASESORES','COLSESCO','VIVE','RUGE']) else 'Ejecutivo'),
        'meta':round(e['meta'],1),'real':round(e['acum'],1),'cump':round(100*e['acum']/e['meta'],1),
        'real_sFV':round(e['acum']-fvadj,1),'cump_sFV':round(100*(e['acum']-fvadj)/e['meta'],1),'fv':fvadj>0,
        'meta_mes':[round(x,1) for x in pm.get(nm,[0]*12)[:9]],'real_mes':[round(x,1) for x in e['real']],
        'meta_anual':round(sum(pm.get(nm,[0]*12)),1)})
rm=np.sum([e['real'] for e in ej],axis=0)
rep['real_mes']=[round(x,1) for x in rm]
mt=sum(e['meta'] for e in ej); rt=sum(e['acum'] for e in ej)
errs=np.array([e['acum']-e['meta'] for e in ej]); errs_s=np.array([e['real_sFV']-e['meta'] for e in rep['ejecutivos']])
rep.update({'meta':round(mt,1),'real':round(rt,1),'cump':round(100*rt/mt,1),'cump_sFV':round(100*(rt-998.613)/mt,1),
  'wape':round(100*np.abs(errs).sum()/mt,1),'wape_sFV':round(100*np.abs(errs_s).sum()/mt,1),
  'mediana_cump':round(float(np.median([e['cump'] for e in rep['ejecutivos']])),1),
  'n_sobre':int(sum(e['cump']>=100 for e in rep['ejecutivos'])),'n_bajo70':int(sum(e['cump']<70 for e in rep['ejecutivos'])),'n':len(ej)})
print({k:v for k,v in rep.items() if k not in('ejecutivos',)})
out['cump']=cum; out['reporte']=rep
json.dump(out,open('res_a.json','w'),ensure_ascii=False)
pd.to_pickle(S,'S.pkl'); pd.to_pickle(bud,'bud.pkl')
