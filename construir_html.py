import json, pandas as pd, datetime
a=json.load(open('res_a.json')); b=json.load(open('res_b.json'))
df=pd.read_pickle('long.pkl')
filas=df.drop_duplicates(['libro','anio','fila']).groupby(['anio','libro']).size().unstack().to_dict('index')
D={**a,**b}
D['filas']={str(k):v for k,v in filas.items()}
D['base2026_sFV']=round(a['car']['TOT_sFV']['js26']+b['fc']['TOT_sFV']['oct_dic']['mediana'],1)
D['controles']=[
 '22 hojas leídas (11 años × 2 libros); 129 meses por serie, de ene-2016 a sep-2026',
 '4.623 filas de datos en el libro empresarial y 655 en el individual',
 '859 y 154 números de contrato; 961 distintos en la unión; 52 presentes en ambos libros',
 '0 filas vacías y 0 valores negativos en las columnas D–O',
 'Columna P (total por contrato) = suma D–O en todas las filas donde existe (0 diferencias)',
 'Fila «TOTAL GENERAL» del libro empresarial = suma de las filas, mes a mes, en los 11 años (diferencia $0)',
 'Fila «TOTAL GENERAL» del libro individual está vacía: el total se reconstruye desde las filas',
 'Totales 2016–2025: $82.156.367.764 (empresarial) y $8.479.077.697 (individual), iguales al avance',
 'Conservación del total: la base larga suma exactamente lo mismo que las hojas de origen',
 'Procedencia por registro: libro, hoja (año) y fila de Excel',
 'Regla R5: no se fusionan ni eliminan filas por compartir número o nombre de contrato',
 'Presupuesto 2026: total general $13.152 M = empresarial $11.731 M + individual $1.411 M + $11,1 M sin desagregar por libro (se usa el total general para la línea)',
 'Reporte por ejecutivo: suma de 23 filas = $10.635 M real y $9.795 M meta (igual al avance)',
 'Nombres de asesores y ejecutivos seudonimizados en las salidas']
D['objetivos']=[
 dict(n=1,t='Base analítica reproducible',antes=90,ahora=100,e='Base larga con procedencia, 14 controles documentados y conciliación con diferencia $0; scripts reejecutables.'),
 dict(n=2,t='Caracterizar 2026 y fijar la regla de atípicos',antes=70,ahora=100,e='Tendencia, peso por mes, STL, atípicos por regla explícita, contratos que explican los picos y regla R7 escrita.'),
 dict(n=3,t='Cumplimiento 2026 a septiembre',antes=55,ahora=90,e='Cumplimiento mensual y acumulado por libro y total, por ejecutivo, con y sin el pago, y desviaciones explicadas por contrato.'),
 dict(n=4,t='Validar, seleccionar métodos y estimar el cierre 2026',antes=30,ahora=75,e='ADF, KPSS, Box-Cox, Ljung-Box; 4 métodos con origen móvil; prueba final; cierre esperado con intervalo.'),
 dict(n=5,t='Proyección mensual 2027 con escenarios',antes=5,ahora=65,e='12 meses con intervalos 80 % y 95 %, escenarios sin pago / con pago / esperado, componentes por libro.')]
D['avance_total']=round(sum(o['ahora'] for o in D['objetivos'])/5)
D['pendientes']=[
 'Obj. 3 · Confirmar con la empresa por qué el reporte y los libros reparten distinto enero y febrero (fecha de pago vs. fecha de registro).',
 'Obj. 4 · Validación prospectiva con los datos reales de octubre–diciembre de 2026 cuando la empresa los entregue.',
 'Obj. 4 · Probar el modelado de intervenciones (Chen y Liu, 1993) para FONCEL, SOMEC y el corte de septiembre.',
 'Obj. 4 · Deflactar con el IPC mensual para reportar crecimiento real.',
 'Obj. 5 · Validar los escenarios y el mes del pago de Fondavivienda con la Dirección Comercial.',
 'Obj. 5 · Contrastar con la consolidación de las plantillas de los ejecutivos (desde el 20-oct-2026) y, si alcanza el tiempo, reconciliación jerárquica.']
D['generado']=datetime.date.today().strftime('%d/%m/%Y')
html=open('plantilla.html',encoding='utf-8').read().replace('__DATA__',json.dumps(D,ensure_ascii=False,default=float))
open('Exepaz_Empresarial_Avance70.html','w',encoding='utf-8').write(html)
print(D['avance_total'],D['base2026_sFV'],len(html)//1024,'KB')
