import pandas as pd, numpy as np, json, warnings, time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
warnings.filterwarnings('ignore')
S=pd.read_pickle('S.pkl'); idx=S['TOT'].index
Z80,Z95=stats.norm.ppf(0.9),stats.norm.ppf(0.975)

def fc_snaive(y,h):
    f=np.array([y[len(y)-12+((i)%12)] for i in range(h)])
    e=y[12:]-y[:-12]; sd=np.std(e,ddof=1); k=np.floor(np.arange(h)/12)+1
    return f,(f-Z80*sd*np.sqrt(k),f+Z80*sd*np.sqrt(k)),(f-Z95*sd*np.sqrt(k),f+Z95*sd*np.sqrt(k))
def fc_actual(y,h):
    g=y[-12:].sum()/y[-24:-12].sum()
    return np.array([y[len(y)-12+(i%12)]*g for i in range(h)]),None,None
def fc_sarima(y,h,ret=False):
    m=SARIMAX(np.log(y),order=(0,1,1),seasonal_order=(0,1,1,12)).fit(disp=False)
    p=m.get_forecast(h); mu=p.predicted_mean; ci8=p.conf_int(alpha=0.2); ci9=p.conf_int(alpha=0.05)
    r=(np.exp(mu),(np.exp(ci8[:,0]),np.exp(ci8[:,1])),(np.exp(ci9[:,0]),np.exp(ci9[:,1])))
    return (r,m) if ret else r
def _ps(y): return pd.Series(np.log(y),index=pd.date_range('2016-01-01',periods=len(y),freq='MS'))
def fc_ets(y,h,ret=False):
    m=ETSModel(_ps(y),error='add',trend='add',damped_trend=True,seasonal='add',seasonal_periods=12).fit(disp=False)
    p=m.get_prediction(start=len(y),end=len(y)+h-1)
    a=p.summary_frame(alpha=0.2); b=p.summary_frame(alpha=0.05)
    r=(np.exp(a['mean'].values),(np.exp(a.pi_lower.values),np.exp(a.pi_upper.values)),(np.exp(b.pi_lower.values),np.exp(b.pi_upper.values)))
    return (r,m) if ret else r
MODELS={'Estacional ingenuo':fc_snaive,'Método actual (replicado)':fc_actual,'ETS(A,Ad,A) log':fc_ets,'SARIMA (0,1,1)(0,1,1)12 log':fc_sarima}

def evaluar(key,origins,end_period):
    y_all=S[key].values; rows=[]
    endi=list(idx).index(pd.Period(end_period,'M'))
    for o in origins:
        oi=list(idx).index(pd.Period(o,'M')); y=y_all[:oi+1]
        h=min(12,endi-oi); 
        if h<1: continue
        scale=np.mean(np.abs(y[12:]-y[:-12]))
        for mn,f in MODELS.items():
            mu,c8,c9=f(y,h); act=y_all[oi+1:oi+1+h]
            for j in range(h):
                rows.append(dict(serie=key,modelo=mn,origen=o,h=j+1,act=act[j],fc=mu[j],scale=scale,
                    in80=float(c8[0][j]<=act[j]<=c8[1][j]) if c8 is not None else np.nan,in95=float(c9[0][j]<=act[j]<=c9[1][j]) if c9 is not None else np.nan))
    return pd.DataFrame(rows)
def anual(d):
    d=d.copy(); full=d.groupby(['serie','modelo','origen']).h.transform('max')==12
    a=d[full].groupby(['serie','modelo','origen']).agg(act=('act','sum'),fc=('fc','sum')).reset_index()
    a['ape']=100*np.abs(a.fc-a.act)/a.act; a['pe']=100*(a.fc-a.act)/a.act
    return a.groupby(['serie','modelo']).agg(APE12=('ape','mean'),sesgo12=('pe','mean'),n12=('ape','size')).reset_index()
def resumen(d):
    d=d.copy(); d['ae']=np.abs(d.act-d.fc); d['se']=(d.act-d.fc)**2; d['sae']=d.ae/d.scale
    g=d.groupby(['serie','modelo']).agg(MAE=('ae','mean'),RMSE=('se',lambda x:np.sqrt(x.mean())),MASE=('sae','mean'),C80=('in80','mean'),C95=('in95','mean'),n=('ae','size')).reset_index()
    return g
t=time.time()
SERIES=['TOT','TOT_sFV','EMP','IND']
dev_orig=[str(p) for p in pd.period_range('2019-12','2023-12',freq='M')]  # 49 orígenes
dev=pd.concat([evaluar(k,dev_orig,'2024-12') for k in SERIES]); print('dev',time.time()-t)
rd=resumen(dev).merge(anual(dev)); print(rd.round(3).to_string())
dev['ae']=np.abs(dev.act-dev.fc); dev['sae']=dev.ae/dev.scale
byh=dev.groupby(['serie','modelo','h']).sae.mean().round(3)
test_orig=[str(p) for p in pd.period_range('2024-12','2026-08',freq='M')]
test=pd.concat([evaluar(k,test_orig,'2026-09') for k in SERIES]); rt=resumen(test).merge(anual(test)); print('PRUEBA'); print(rt.round(3).to_string())

# diagnóstico
diag={}
for k in ['TOT','TOT_sFV','EMP','IND']:
    y=np.log(S[k].values); d1=np.diff(y); ds=y[12:]-y[:-12]; dd=np.diff(ds)
    def t_(x):
        a=adfuller(x,autolag='AIC'); kp=kpss(x,regression='c',nlags='auto')
        return {'adf_p':round(a[1],4),'kpss_p':round(kp[1],4)}
    lam=stats.boxcox(S[k].values)[1]
    (_,m)=fc_sarima(S[k].values[:list(idx).index(pd.Period('2024-12','M'))+1],1,ret=True)
    lb=acorr_ljungbox(m.resid[13:],lags=[12,24],return_df=True)
    diag[k]={'nivel_log':t_(y),'dif_estacional':t_(ds),'dif_reg_estac':t_(dd),'boxcox_lambda':round(lam,3),
             'ljung12_p':round(lb.lb_pvalue.iloc[0],4),'ljung24_p':round(lb.lb_pvalue.iloc[1],4),
             'theta':round(m.params[0],3),'Theta':round(m.params[1],3),'aic':round(m.aic,1)}
    print(k,diag[k])

# ---- pronóstico final con SARIMA (seleccionado) sobre datos completos
def sim_paths(key,h=15,N=5000,seed=1):
    y=S[key].values; m=SARIMAX(np.log(y),order=(0,1,1),seasonal_order=(0,1,1,12)).fit(disp=False)
    sims=m.simulate(nsimulations=h,repetitions=N,anchor='end',rng=np.random.default_rng(seed))
    return np.exp(np.squeeze(sims)).T  # N x h
def ets_paths(key,h=15,N=5000):
    y=S[key].values; m=ETSModel(_ps(y),error='add',trend='add',damped_trend=True,seasonal='add',seasonal_periods=12).fit(disp=False)
    sims=m.simulate(nsimulations=h,repetitions=N,anchor='end',random_errors='bootstrap',rng=np.random.default_rng(2))
    return np.exp(np.asarray(sims)).T
fut=[str(p) for p in pd.period_range('2026-10','2027-12',freq='M')]
fc={}
for k in ['TOT_sFV','EMP_sFV','IND_sFV','TOT']:
    P=sim_paths(k)
    q=lambda a,qq: [round(float(x),2) for x in np.percentile(a,qq,axis=0)]
    fc[k]={'media':[round(float(x),2) for x in P.mean(0)],'mediana':q(P,50),'p10':q(P,10),'p90':q(P,90),'p2_5':q(P,2.5),'p97_5':q(P,97.5),
           'oct_dic':{'mediana':round(float(np.median(P[:,:3].sum(1))),1),'p10':round(float(np.percentile(P[:,:3].sum(1),10)),1),'p90':round(float(np.percentile(P[:,:3].sum(1),90)),1),'p2_5':round(float(np.percentile(P[:,:3].sum(1),2.5)),1),'p97_5':round(float(np.percentile(P[:,:3].sum(1),97.5)),1)},
           'anio2027':{'mediana':round(float(np.median(P[:,3:].sum(1))),1),'p10':round(float(np.percentile(P[:,3:].sum(1),10)),1),'p90':round(float(np.percentile(P[:,3:].sum(1),90)),1),'p2_5':round(float(np.percentile(P[:,3:].sum(1),2.5)),1),'p97_5':round(float(np.percentile(P[:,3:].sum(1),97.5)),1)},
           'paths_anual':None}
    if k=='TOT_sFV':
        np.save('paths_tot_sfv.npy',P)
    print(k,fc[k]['oct_dic'],fc[k]['anio2027'])
# ETS para contraste de 2027
Pe=ets_paths('TOT_sFV'); fc['TOT_sFV_ETS']={'mediana':[round(float(x),2) for x in np.median(Pe,0)],'anio2027':round(float(np.median(Pe[:,3:].sum(1))),1)}
print('ETS 2027',fc['TOT_sFV_ETS']['anio2027'])
# cierre 2026: real ene-sep (con FV) + pronóstico oct-dic sin FV
bud=pd.read_pickle('bud.pkl'); cierre={}
for lib,k in [('TOT','TOT_sFV'),('EMP','EMP_sFV'),('IND','IND_sFV')]:
    P=sim_paths(k); real=S[lib][[p for p in idx if p.year==2026]].sum(); real_s=S[k][[p for p in idx if p.year==2026]].sum()
    tot=real+P[:,:3].sum(1); tot_s=real_s+P[:,:3].sum(1); B=sum(bud[lib])
    pc=lambda a:[round(float(np.percentile(a,q)),1) for q in (2.5,10,50,90,97.5)]
    cierre[lib]={'real_js':round(real,1),'presu_anual':round(B,1),'presu_od':round(sum(bud[lib][9:]),1),
       'od':pc(P[:,:3].sum(1)),'total':pc(tot),'cump':pc(100*tot/B),'cump_sFV':pc(100*tot_s/B),'p_cumple':round(float((tot>=B).mean())*100,1),'p_cumple_sFV':round(float((tot_s>=B).mean())*100,1)}
    print(lib,cierre[lib])
res={'eval_dev':rd.round(4).to_dict('records'),'eval_test':rt.round(4).to_dict('records'),
     'mase_h':{f'{s}|{m}':[float(byh[(s,m,h)]) for h in range(1,13)] for s in SERIES for m in MODELS},
     'diag':diag,'fc':fc,'fut':fut,'cierre':cierre}
json.dump(res,open('res_b.json','w'),ensure_ascii=False,default=float)
