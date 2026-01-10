import ssl 
# --- EL PARCHE MÁGICO ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ------------------------

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
import requests
import numpy as np
from scipy.optimize import minimize
import io
import xlsxwriter


# 1. Configuración de la página
st.set_page_config(layout='wide', page_title='Dimabe Investments')
st.title('Gestión de Inversiones')
st.markdown('---')

# 2. Funciones de carga de datos 
@st.cache_data
def tickers_sp500():
    """Descarga la lista de S&P 500 desde Wikipedia simulando ser un navegador"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # ESTA ES LA CLAVE: Le decimos que somos un navegador Mozilla
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        
        # 1. Hacemos la petición con los headers
        respuesta = requests.get(url, headers=headers)
        
        # 2. Leemos la tabla desde el texto de la respuesta
        html = pd.read_html(respuesta.text)
        
        df = html[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers] # Arreglo para BRK.B
        return tickers
    except Exception as e:
        st.error(f"Error cargando S&P 500: {e}")
        return []

@st.cache_data
def descargar_y_convertir_a_clp(start_date):
    """
    Descarga TODO el S&P 500 completo + Chile + Cripto y convierte a CLP.
    Versión 'Full Market': Puede tardar un poco más en cargar, pero tiene toda la data.
    """
    # A. Definimos el Universo
    # LISTA IPSA EXPANDIDA (Principales acciones de Chile)
    tickers_chile = [
        'SQM-B.SN', 'CHILE.SN', 'BSANTANDER.SN', 'COPEC.SN', 'CENCOSUD.SN', 'FALABELLA.SN',
        'CMPC.SN', 'ENELAM.SN', 'VAPORES.SN', 'CAP.SN', 'ANDINA-B.SN', 'CCU.SN',
        'AGUAS-A.SN', 'BCI.SN', 'CENCOSHOPP.SN', 'COLBUN.SN', 'CONCHATORO.SN',
        'ENTEL.SN', 'IAM.SN', 'ILC.SN', 'LTM.SN', 'MALLPLAZA.SN', 'PARAUCO.SN',
        'QUINENCO.SN', 'RIPLEY.SN', 'SMU.SN', 'SONDA.SN', 'ENELCHILE.SN'
    ]
    
    # 1. Obtenemos la lista COMPLETA de Wikipedia (500+ acciones)
    tickers_usa = tickers_sp500()
    
    # 2. Aseguramos que estén los "Gigantes" y el Benchmark (SPY)
    # (A veces Wikipedia tarda en actualizar o usa nombres raros, así que aseguramos estos)
    indispensables = ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'SPY']
    for t in indispensables:
        if t not in tickers_usa:
            tickers_usa.append(t)
    
    tickers_bonos = ['SHY', 'TLT', 'LQD', 'BND']
    tickers_crypto = ['BTC-USD', 'ETH-USD']
    
    # Juntamos todo (Usamos set para eliminar duplicados si los hubiera)
    todos_tickers = list(set(tickers_chile + tickers_usa + tickers_bonos + tickers_crypto + ['CLP=X']))
    
    # B. Descarga Masiva
    
    data = yf.download(todos_tickers, start=start_date)['Close']
    
    # Limpieza de zona horaria
    data.index = data.index.tz_localize(None)
    
    # C. Relleno del Dólar
    dolar = data['CLP=X']
    dolar = dolar.ffill().bfill()
    
    data_clp = data.copy()
    
    # D. Conversión Masiva a CLP
    # Optimizamos para que no sea lento con 500 acciones
    cols_a_convertir = [col for col in data.columns if not col.endswith('.SN') and col != 'CLP=X']
    
    # Vectorización (Más rápido que un loop for normal)
    data_clp[cols_a_convertir] = data[cols_a_convertir].multiply(dolar, axis=0)
    
    # Limpieza final
    data_clp = data_clp.ffill()
    data_clp = data_clp.dropna(axis=1, how='all') # Borra las que fallaron
    
    # Filtro de Calidad: Borrar acciones que tengan menos del 90% de datos
    # (Para evitar acciones nuevas que rompan los gráficos)
    limit = len(data_clp) * 0.9
    data_clp = data_clp.dropna(axis=1, thresh=limit)

    return data_clp, dolar

def calcular_rsi(data, window=14):
    '''Calcula el Relative Strength Index (RSI) para una serie de datos.'''
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def optimizar_portafolio(datos_seleccionados):
    """
    Calcula SOLO los 2 extremos de la Frontera Eficiente:
    1. Piso: Mínima Volatilidad (Conservador).
    2. Techo: Máximo Retorno con Diversificación (Agresivo).
    """
    retornos = datos_seleccionados.pct_change().dropna()
    
    if retornos.empty:
        n = len(datos_seleccionados.columns)
        return np.full(n, 1/n), np.full(n, 1/n) # Solo devolvemos 2

    n_activos = len(datos_seleccionados.columns)
    
    # Funciones Objetivo
    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum(retornos.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(retornos.cov() * 252, weights)))
        return np.array([ret, vol])

    def minimize_volatility(weights): return get_ret_vol_sr(weights)[1]
    def maximize_return(weights): return -get_ret_vol_sr(weights)[0]

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = n_activos * [1. / n_activos,]

    # 1. CONSERVADOR (Piso) - Permitimos 100% en Bonos
    bounds_cons = tuple((0.0, 1.0) for _ in range(n_activos))
    opt_cons = minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds_cons, constraints=cons)
    
    # 2. AGRESIVO (Techo) - Limitamos al 30% por acción
    bounds_agg = tuple((0.0, 0.30) for _ in range(n_activos)) 
    opt_agg = minimize(maximize_return, init_guess, method='SLSQP', bounds=bounds_agg, constraints=cons)

    # Solo devolvemos los dos extremos
    return opt_cons.x, opt_agg.x

# 3. Sidebar y Carga
st.sidebar.header('Panel de Control')
fecha_inicio = st.sidebar.date_input('Inicio de Análisis',
pd.to_datetime('2020-01-01'))

with st.spinner("⏳ Descargando datos masivos del S&P 500 y Chile..."):
    df_precios_clp, df_dolar = descargar_y_convertir_a_clp(fecha_inicio)

if 'descarga_ok' not in st.session_state:
    st.toast("Datos actualizados correctamente", icon="✅")
    st.session_state['descarga_ok'] = True

## Mostrar dólar actual
precio_dolar_hoy = df_dolar.iloc[-1]
st.sidebar.metric('Dólar Observado (Hoy)', f'${precio_dolar_hoy:,.0f} CLP')

# 4. Estructura de Pestañas
tab1, tab2, tab3 = st.tabs(['Top de Acciones', 'Análisis Técnico', 'Inversiones'])

# =====================================
# PESTAÑA 1: RANKING DE ACCIONES
# =====================================

with tab1:
    st.header('Ranking de Mercado (en CLP)')
    st.markdown('Descubre qué acciones están ganando o perdiendo valor para un inversionista chileno.')

    ## Selector de ventana de tiempo
    ventana = st.radio('Ordenar por Rendimiento:', ['Última Semana', 'Último Mes', 'Año Actual (YTD)'], horizontal=True)

    ## Calcular retornos según la ventana elegida 
    if ventana == 'Última Semana':
        dias = 5
        retornos = df_precios_clp.pct_change(periods=dias).iloc[-1]
    elif ventana == 'Útlimo Mes':
        dias = 20
        retornos = df_precios_clp.pct_change(periods=dias).iloc[-1]
    else: # YTD (Desde inicio de año)
        fecha_ini_anio = f'{datetime.now().year}-01-01'
        df_ytd = df_precios_clp[df_precios_clp.index >= fecha_ini_anio] # Filtrar desde el 1 de Enero
        if len(df_ytd) > 0:
            retorno_total = (df_ytd.iloc[-1] / df_ytd.iloc[0]) - 1
            retornos = retorno_total
        else:
            retornos = df_precios_clp.pct_change(periods=252).iloc[-1]
    
    ## Convertir a DataFrame
    ranking = retornos.to_frame(name='Retorno')
    ranking = ranking.drop('CLP=X', errors='ignore') # Se saca el dólar del ranking
    ranking['Retorno %'] = ranking['Retorno'].apply(lambda x: f'{x:.2%}')

    ## Top Ganadores y Perdedores 
    col_win, col_loss = st.columns(2)

    with col_win:
        st.subheader('Top 10 Ganadores')
        top_winners = ranking.sort_values('Retorno', ascending=False).head(10)

        ### Colores
        st.dataframe(
            top_winners[['Retorno %']],
            use_container_width=True
        )
    with col_loss:
        st.subheader('Top 10 Perdedores')
        top_losers = ranking.sort_values('Retorno', ascending=True).head(10)
        st.dataframe(
            top_losers[['Retorno %']],
            use_container_width=True
        )
    
    ## Gráfico de Barras del Top Ganadores
    fig_ranking = px.bar(
        top_winners,
        y='Retorno',
        x=top_winners.index,
        title=f'Líderes del Mercado ({ventana}) - Retorno en CLP',
        color='Retorno',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig_ranking, use_container_width=True)

# =====================================
# PESTAÑA 2: 
# =====================================
    
with tab2:
    st.header('Análisis de Precios')

    # 1. Selectores
    col_sel1, col_sel2, col_sel3 = st.columns(3)

    with col_sel1:
        ## Lista de activos disponibles en la data descargada
        lista_activos = df_precios_clp.columns.tolist()
        activo_elegido = st.selectbox('Selecciona Activo:', lista_activos, 
                                      index=lista_activos.index('SQM-B.SN') if 'SQM-B.SN' in lista_activos else 0)
    
    with col_sel2:
        sma_corta = st.number_input('Media Movil Corta (Días)', value=20, min_value=5)
    
    with col_sel3:
        sma_larga = st.number_input('Media Movil Larga (Días)', value=50, min_value=10)
    
    # 2. Preparación de datos para el activo elegido
    if activo_elegido:
        ## Extraemos la serie de precios del activo
        serie_precios = df_precios_clp[activo_elegido]

        ## Calcular indicadores
        sma_s = serie_precios.rolling(window=sma_corta).mean()
        sma_l = serie_precios.rolling(window=sma_larga).mean()
        rsi = calcular_rsi(serie_precios)

        ## Precio actual y variación
        precio_actual = serie_precios.iloc[-1]
        precio_ayer = serie_precios.iloc[-2]
        delta = precio_actual - precio_ayer
        delta_pct = delta / precio_ayer

        ## Métricas grandes
        c1, c2, c3 = st.columns(3)
        c1.metric('Precio Actual (CLP)', f'${precio_actual:,.0f}',
                  f'{delta_pct:.2%}')
        c2.metric('RSI (14)', f'{rsi.iloc[-1]:.1f}', delta_color='off')

        ## Interpretación rápida del RSI
        valor_rsi = rsi.iloc[-1]
        estado_rsi = 'Neutro'
        if valor_rsi > 70: estado_rsi = 'Sobrecomprado (Posible Caída)'
        elif valor_rsi < 30: estado_rsi = 'Sobrevendido (Oportunidad)'
        c3.write(f'**Señal RSI:** {estado_rsi}')

        ## Gráfico principal (Precio + SMA)
        fig_price = px.line(serie_precios, title=f'Evolución de Precio: {activo_elegido}')
        fig_price.update_traces(line=dict(color='#1f77b4', width=3), name='Precio')

        ## Agregar las SMAs
        fig_price.add_scatter(x=serie_precios.index, y=sma_s, mode='lines', 
                              name=f'SMA {sma_corta}', line=dict(color='#ff7f0e', width=2))
        fig_price.add_scatter(x=serie_precios.index, y=sma_l, mode='lines', 
                              name=f'SMA {sma_larga}', line=dict(color='#2ca02c', width=2))
        
        fig_price.update_layout(yaxis_title='Precio en CLP', xaxis_title='Fecha')
        
        st.plotly_chart(fig_price, use_container_width=True)

        ## Gráfico secundario (RSI)
        st.markdown('#### Índice de Fuerza Relativa (RSI)')
        fig_rsi = px.line(rsi, title='Momentum (RSI)')

        ## Zonas Clave (70 y 30)
        fig_rsi.add_hline(y=70, line_dash='dash', line_color='red', 
                          annotation_text='Sobrecompra')
        fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', 
                          annotation_text='Sobreventa')
        fig_rsi.update_yaxes(range=[0, 100])

        st.plotly_chart(fig_rsi, use_container_width=True)

# =====================================
# PESTAÑA 3:
# =====================================


with tab3:
    st.header("🤖 Robo-Advisor: Tu Estrategia Personalizada")
    
    col_izq, col_der = st.columns([1, 2])
    
    with col_izq:
        st.subheader("1. Configura tu Perfil")
        monto = st.number_input("💰 Monto a Invertir (CLP)", value=1000000, step=100000)
        riesgo = st.slider("🎚️ Nivel de Riesgo (1=Conservador, 10=Agresivo)", 1, 10, 5)
        
        st.markdown("---")
        st.subheader("2. Selección de Activos")
        
        # --- CEREBRO AUTOMÁTICO ---
        st.caption("🤖 El sistema ha seleccionado los ganadores del último mes + seguridad.")
        
        bonos_seguros = ['SHY', 'LQD'] 
        criptos = ['BTC-USD', 'ETH-USD']
        
        # Calculamos Momentum (20 días)
        rendimiento_mes = df_precios_clp.pct_change(periods=20).iloc[-1].dropna()
        excluir = ['CLP=X'] + bonos_seguros + criptos
        solo_acciones = rendimiento_mes.drop(index=excluir, errors='ignore')
        top_5_acciones = solo_acciones.sort_values(ascending=False).head(5).index.tolist()
        
        seleccion_inteligente = bonos_seguros + top_5_acciones + ['BTC-USD']
        seleccion_final = [x for x in seleccion_inteligente if x in df_precios_clp.columns]

        activos_elegidos = st.multiselect(
            "Canasta de Inversión Sugerida:", 
            df_precios_clp.columns, 
            default=seleccion_final
        )
        
        btn_optimizar = st.button("🚀 Generar Portafolio Óptimo")

    with col_der:
        if btn_optimizar and len(activos_elegidos) > 2:
            with st.spinner("Ejecutando simulación de Montecarlo simplificada y optimizando..."):
                datos_opt = df_precios_clp[activos_elegidos]
                
                # 1. OPTIMIZACIÓN MATEMÁTICA (RECIBE 2 EXTREMOS)
                w_cons, w_agg = optimizar_portafolio(datos_opt)

                # 2. MEZCLA LINEAL SEGÚN SLIDER (1 al 10)
                # Factor va de 0.0 (Nivel 1) a 1.0 (Nivel 10)
                factor = (riesgo - 1) / 9.0
                
                # Fórmula de Mezcla: (Parte Segura) + (Parte Riesgosa)
                pesos_finales = (1 - factor) * w_cons + factor * w_agg
                
                # Etiqueta Dinámica
                if riesgo <= 3: etiqueta = "Perfil Conservador"
                elif riesgo <= 7: etiqueta = "Perfil Moderado"
                else: etiqueta = "Perfil Agresivo"
                
                # 3. DATAFRAME DE RESULTADOS
                df_res = pd.DataFrame({
                    "Activo": activos_elegidos,
                    "Peso": pesos_finales,
                    "Inversión (CLP)": pesos_finales * monto
                })
                df_res = df_res[df_res["Peso"] > 0.01].sort_values("Peso", ascending=False)
                
                # 4. CÁLCULOS DE RENDIMIENTO Y RIESGO
                retornos_activos = datos_opt.pct_change().mean() * 252
                retorno_portafolio = np.sum(retornos_activos * pesos_finales)
                
                cov_matrix = datos_opt.pct_change().cov() * 252
                varianza_portafolio = np.dot(pesos_finales.T, np.dot(cov_matrix, pesos_finales))
                riesgo_portafolio = np.sqrt(varianza_portafolio)
                
                # --- CÁLCULO DEL VALUE AT RISK (VaR) 95% ---
                z_score_95 = 1.645
                var_percent = z_score_95 * riesgo_portafolio
                var_monto = monto * var_percent

                # --- VISUALIZACIÓN DE RESULTADOS ---
                st.success(f"✅ Estrategia Generada: **{etiqueta}** (Nivel {riesgo}/10)")
                
                # A. KPIs PRINCIPALES
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Monto Inversión", f"${monto:,.0f}")
                kpi2.metric("Retorno Esp. Anual", f"{retorno_portafolio:.1%}")
                kpi3.metric("Riesgo (Volatilidad)", f"{riesgo_portafolio:.1%}")
                kpi4.metric("Value at Risk (95%)", f"${var_monto:,.0f}", help="Pérdida máxima esperada en un año malo (con 95% de confianza).")

                # B. GRÁFICO DE DONA
                fig_pie = px.pie(df_res, values="Inversión (CLP)", names="Activo", title="Distribución de Activos", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # --- OPCIÓN 1 - MATRIZ DE CORRELACIÓN ---
                with st.expander("📊 Ver Matriz de Correlación (Detector de Diversificación)", expanded=False):
                    st.write("¿Tus activos se mueven juntos? (Rojo = Sí / Azul = No)")
                    st.caption("Busca colores AZULES para mejor diversificación.")
                    corr_matrix = datos_opt.corr()
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        color_continuous_scale="RdBu_r", # Rojo a Azul invertido
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                st.markdown("---")

                # --- OPCIÓN 3 - VALUE AT RISK EXPLICADO ---
                st.subheader("📉 Análisis de Riesgo Extremo (VaR)")
                st.info(
                    f"**Interpretación del VaR:** Según la volatilidad histórica de tu portafolio, existe un **95% de probabilidad** "
                    f"de que tu pérdida anual NO supere los **${var_monto:,.0f}**. "
                    f"Solo en un escenario de crisis (el 5% restante) perderías más que eso."
                )
                
                # Tabla Comparativa de Riesgos
                st.write("**Comparativa de Riesgo:**")
                
                def calcular_var_rapido(pesos):
                    vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
                    return monto * z_score_95 * vol

                var_cons = calcular_var_rapido(w_cons)
                var_agg = calcular_var_rapido(w_agg)
                
                df_var = pd.DataFrame({
                    "Perfil": ["🛡️ Conservador (Piso)", "🚀 Agresivo (Techo)", "✅ TU ESTRATEGIA"],
                    "Pérdida Máx. Estimada (VaR 95%)": [var_cons, var_agg, var_monto],
                    "Volatilidad Anual": [
                        np.sqrt(np.dot(w_cons.T, np.dot(cov_matrix, w_cons))),
                        np.sqrt(np.dot(w_agg.T, np.dot(cov_matrix, w_agg))),
                        riesgo_portafolio
                    ]
                })
                
                st.dataframe(
                    df_var.style.format({
                        "Pérdida Máx. Estimada (VaR 95%)": "${:,.0f}", 
                        "Volatilidad Anual": "{:.1%}"
                    }),
                    use_container_width=True
                )

                st.markdown("---")

                # TABLA DE COMPRA
                st.subheader("📋 1. Orden de Compra Sugerida")
                st.dataframe(df_res.style.format({"Peso": "{:.1%}", "Inversión (CLP)": "${:,.0f}"}), use_container_width=True)

                # --- NUEVO: BOTÓN DE DESCARGA EXCEL (.XLSX) ---
                # 1. Crear un buffer de memoria (un archivo virtual)
                buffer = io.BytesIO()
                
                # 2. Usar Pandas para escribir el Excel en ese buffer
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_res.to_excel(writer, index=False, sheet_name='Mi Portafolio')
                    
                    # (Opcional) Formato bonito: Ajustar ancho de columnas
                    workbook = writer.book
                    worksheet = writer.sheets['Mi Portafolio']
                    format_money = workbook.add_format({'num_format': '$#,##0'})
                    format_pct = workbook.add_format({'num_format': '0.0%'})
                    
                    worksheet.set_column('B:B', 12, format_pct) # Columna Peso
                    worksheet.set_column('C:C', 15, format_money) # Columna Inversión

                # 3. Botón de descarga leyendo del buffer
                st.download_button(
                    label="📥 Descargar Excel (.xlsx)",
                    data=buffer.getvalue(),
                    file_name=f"Estrategia_{etiqueta.replace(' ', '_')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

                st.markdown("---")
                
                # --- TABLA DE PROYECCIONES TEMPORALES (RECUPERADA ✅) ---
                st.subheader("📅 2. Proyección de Ganancias (Estimado)")
                
                tasas = {
                    "Diario": retorno_portafolio / 252,
                    "Semanal": retorno_portafolio / 52,
                    "Mensual": retorno_portafolio / 12,
                    "Semestral": retorno_portafolio / 2,
                    "Anual": retorno_portafolio
                }
                
                data_proyeccion = []
                for periodo, tasa in tasas.items():
                    ganancia = monto * tasa
                    data_proyeccion.append({
                        "Periodo": periodo,
                        "Retorno Estimado %": f"{tasa:.2%}",
                        "Ganancia Estimada ($)": f"${ganancia:,.0f}"
                    })
                
                df_proy = pd.DataFrame(data_proyeccion)
                st.table(df_proy)
                st.caption("Nota: Rentabilidades pasadas no garantizan rentabilidades futuras.")

                st.markdown("---")
                        
                # --- BACKTESTING (LA PRUEBA DE FUEGO) ---
                st.subheader("🧪 Backtesting: Tu Estrategia vs. El Mercado")
                st.caption("Si hubieras invertido $1.000.000 hace un año, ¿cómo te habría ido comparado con el S&P 500?")

                # 1. Preparamos los datos del Benchmark (SPY)
                if 'SPY' in df_precios_clp.columns:
                    benchmark = df_precios_clp['SPY']
                else:
                    benchmark = datos_opt.mean(axis=1)

                # 2. Construimos TU Portafolio Histórico
                retornos_hist_ponderados = (datos_opt.pct_change() * pesos_finales).sum(axis=1)

                # 3. Normalizamos a Base 100
                ventan_dias = 252
                if len(retornos_hist_ponderados) > ventan_dias:
                    historia_portafolio = (1 + retornos_hist_ponderados.tail(ventan_dias)).cumprod() * monto
                    
                    historia_benchmark = benchmark.pct_change().tail(ventan_dias)
                    historia_benchmark = (1 + historia_benchmark).cumprod() * monto
                    
                    df_backtest = pd.DataFrame({
                        "Tu Estrategia 🤖": historia_portafolio,
                        "S&P 500 (Benchmark) 🇺🇸": historia_benchmark
                    })
                    
                    fig_back = px.area(
                        df_backtest, 
                        title="Simulación Histórica (1 Año)",
                        color_discrete_map={"Tu Estrategia 🤖": "#00CC96", "S&P 500 (Benchmark) 🇺🇸": "#636EFA"}
                    )
                    st.plotly_chart(fig_back, use_container_width=True)
                    
                    resultado_tu = historia_portafolio.iloc[-1]
                    resultado_bm = historia_benchmark.iloc[-1]
                    diferencia = resultado_tu - resultado_bm
                    
                    if diferencia > 0:
                        st.success(f"🏆 ¡Ganaste! Tu estrategia habría generado **${diferencia:,.0f}** más que el S&P 500.")
                    else:
                        st.error(f"⚠️ El Mercado ganó esta vez. Tu estrategia habría generado **${abs(diferencia):,.0f}** menos (pero quizás con menos riesgo).")
                else:
                    st.warning("No hay suficiente historial para hacer un Backtesting de 1 año completo.")

        # --- MENSAJES DE ERROR AL FINAL ---
        elif btn_optimizar:
            st.warning("⚠️ Selecciona al menos 3 activos.")
        else:
            st.info("👈 Configura tu perfil y presiona 'Generar'.")