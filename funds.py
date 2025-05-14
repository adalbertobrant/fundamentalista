import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import time
import matplotlib.pyplot as plt
import io
from PIL import Image
import requests
from functools import lru_cache

# Configuração para gráficos matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# =============================================
# FUNÇÕES DE CACHE E BUSCA DE DADOS
# =============================================

@st.cache_data(ttl=3600)  # Cache válido por 1 hora
def get_stock_data(tickers_list):
    return coletar_dados_paralelo(tickers_list)

@lru_cache(maxsize=500)
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception as e:
        st.warning(f"Erro ao buscar {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def get_stock_history(ticker, period="1y"):
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception as e:
        st.warning(f"Erro ao buscar histórico de {ticker}: {str(e)}")
        return pd.DataFrame()

# =============================================
# FUNÇÕES DE CLASSIFICAÇÃO
# =============================================

def classifica_graham(pl, pvp):
    if pl and pvp:
        score = pl * pvp
        return "Barata" if score < 22.5 else "Cara"
    return "Indefinido"

def classifica_greenblatt(pl, roe):
    if pl and roe:
        return "Barata" if pl < 15 and roe > 0.15 else "Cara"
    return "Indefinido"

def destacar_linhas(row):
    if row["Graham"] == "Barata" and row["Magic"] == "Barata":
        return ['background-color: #d4edda; color: black'] * len(row)
    return [''] * len(row)

# =============================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# =============================================

def processar_ticker(ticker):
    try:
        info = get_ticker_info(ticker)
        hist = get_stock_history(ticker, period="5d")
        last_close = hist['Close'].iloc[-1] if not hist.empty else None
        
        pl = info.get("trailingPE")
        pvp = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        
        return {
            "Ticker": ticker,
            "Preço": last_close,
            "P/L": pl,
            "P/VP": pvp,
            "ROE": roe,
            "Graham": classifica_graham(pl, pvp),
            "Magic": classifica_greenblatt(pl, roe)
        }
    except Exception as e:
        return {
            "Ticker": ticker, 
            "Preço": None, 
            "P/L": None, 
            "P/VP": None, 
            "ROE": None, 
            "Graham": "Indefinido", 
            "Magic": "Indefinido", 
            "Erro": str(e)
        }

def coletar_dados_paralelo(tickers):
    resultados = []
    lote_size = 10
    total_lotes = (len(tickers) + lote_size - 1) // lote_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(total_lotes):
        start_idx = i * lote_size
        end_idx = min((i + 1) * lote_size, len(tickers))
        lote_atual = tickers[start_idx:end_idx]
        
        status_text.text(f"Processando lote {i+1}/{total_lotes} ({start_idx+1}-{end_idx} de {len(tickers)} tickers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=lote_size) as executor:
            futures = {executor.submit(processar_ticker, ticker): ticker for ticker in lote_atual}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    resultados.append(future.result())
                except Exception as e:
                    ticker = futures[future]
                    resultados.append({"Ticker": ticker, "Erro": str(e)})
        
        progress_bar.progress((i + 1) / total_lotes)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(resultados)

# =============================================
# FUNÇÕES DE ANÁLISE TÉCNICA
# =============================================

def calcular_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcular_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def plot_price_chart(ticker, periodo="1y"):
    hist = get_stock_history(ticker, period=periodo)
    if hist.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(hist.index, hist['Close'], label='Preço de Fechamento', color='blue')
    ax1.set_title(f'Histórico de Preços - {ticker}')
    ax1.set_ylabel('Preço')
    ax1.grid(True)
    ax1.legend()
    
    ax2.bar(hist.index, hist['Volume'], color='purple', alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Data')
    ax2.grid(True)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def plot_technical_indicators(ticker, periodo="6mo"):
    hist = get_stock_history(ticker, period=periodo)
    if hist.empty:
        return None, None
    
    # RSI
    rsi = calcular_rsi(hist['Close'])
    fig_rsi = plt.figure(figsize=(10, 4))
    plt.plot(hist.index, rsi, label='RSI', color='blue')
    plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    plt.fill_between(hist.index, 30, rsi.where(rsi < 30), color='green', alpha=0.3)
    plt.fill_between(hist.index, 70, rsi.where(rsi > 70), color='red', alpha=0.3)
    plt.title(f'RSI - {ticker}')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    buf_rsi = io.BytesIO()
    plt.savefig(buf_rsi, format='png')
    buf_rsi.seek(0)
    plt.close(fig_rsi)
    
    # MACD
    macd, signal, histogram = calcular_macd(hist['Close'])
    fig_macd = plt.figure(figsize=(10, 4))
    plt.plot(hist.index, macd, label='MACD', color='blue')
    plt.plot(hist.index, signal, label='Signal Line', color='red')
    plt.bar(hist.index, histogram, label='Histogram', color='green', alpha=0.5)
    plt.title(f'MACD - {ticker}')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    buf_macd = io.BytesIO()
    plt.savefig(buf_macd, format='png')
    buf_macd.seek(0)
    plt.close(fig_macd)
    
    return buf_rsi, buf_macd

# =============================================
# FUNÇÕES DE INTERFACE
# =============================================

def mostrar_detalhes_empresa(ticker):
    st.subheader(f"Detalhes da Empresa: {ticker}")
    
    tab1, tab2, tab3 = st.tabs(["Informações Básicas", "Dados Fundamentalistas", "Análise Técnica"])
    info = get_ticker_info(ticker)
    print(info.keys())
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            logo_url = info.get('logo_url', '')
            if logo_url:
                try:
                    response = requests.get(logo_urlheaders={'User-Agent': 'Mozilla/5.0'})
                    if response.status_code == 200:
                        st.image(Image.open(io.BytesIO(response.content)), width=150)
                except Exception:
                    st.info("Logo não disponível")
        
        with col2:
            st.write(f"**Nome:** {info.get('longName', 'N/A')}")
            st.write(f"**Setor:** {info.get('sector', 'N/A')}")
            st.write(f"**Indústria:** {info.get('industry', 'N/A')}")
            st.write(f"**Site:** {info.get('website', 'N/A')}")
            st.write(f"**País:** {info.get('country', 'N/A')}")
        
        st.subheader("Sobre a Empresa")
        st.write(info.get('longBusinessSummary', 'Informação não disponível'))
        
        if info.get('companyOfficers'):
            st.subheader("Principais Executivos")
            for officer in info.get('companyOfficers', [])[:5]:
                st.write(f"**{officer.get('title', 'N/A')}:** {officer.get('name', 'N/A')}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Indicadores de Valor")
            st.write(f"**Preço Atual:** {info.get('currentPrice', 'N/A')}")
            st.write(f"**P/L:** {info.get('trailingPE', 'N/A')}")
            st.write(f"**P/VP:** {info.get('priceToBook', 'N/A')}")
            st.write(f"**Dividend Yield:** {info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")
        
        with col2:
            st.subheader("Indicadores de Rentabilidade")
            st.write(f"**ROE:** {info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A")
            st.write(f"**ROA:** {info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else "N/A")
            st.write(f"**Margem EBITDA:** {info.get('ebitdaMargins', 0) * 100:.2f}%" if info.get('ebitdaMargins') else "N/A")
        
        st.subheader("Histórico de Preços")
        periodo = st.selectbox("Selecione o período:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key=f"period_{ticker}")
        price_chart = plot_price_chart(ticker, periodo)
        if price_chart:
            st.image(price_chart)
    
    with tab3:
        st.subheader("Indicadores Técnicos")
        periodo_tecnico = st.selectbox("Selecione o período:", ["1mo", "3mo", "6mo", "1y"], index=2, key=f"tech_period_{ticker}")
        
        rsi_chart, macd_chart = plot_technical_indicators(ticker, periodo_tecnico)
        
        if rsi_chart:
            st.subheader("RSI (Relative Strength Index)")
            st.image(rsi_chart)
        
        if macd_chart:
            st.subheader("MACD (Moving Average Convergence Divergence)")
            st.image(macd_chart)

# =============================================
# LISTAS DE TICKERS
# =============================================

IBOV_TICKERS = [
    "ALOS3.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AZUL4.SA", 
    "AZZA3.SA", "B3SA3.SA", "BBSE3.SA", "BBDC3.SA", "BBDC4.SA", 
    "BRAP4.SA", "BBAS3.SA", "BRKM5.SA", "BRAV3.SA", "BRFS3.SA", 
    "BPAC11.SA", "CXSE3.SA", "CRFB3.SA", "CMIG4.SA", "COGN3.SA", 
    "CPLE6.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA", "CVCB3.SA", 
    "CYRE3.SA", "DIRR3.SA", "ELET3.SA", "ELET6.SA", "EMBR3.SA", 
    "ENGI11.SA", "ENEV3.SA", "EGIE3.SA", "EQTL3.SA", "FLRY3.SA", 
    "GGBR4.SA", "GOAU4.SA", "NTCO3.SA", "HAPV3.SA", "HYPE3.SA", 
    "IGTI11.SA", "IRBR3.SA", "ISAE4.SA", "ITSA4.SA", "ITUB4.SA", 
    "JBSS3.SA", "KLBN11.SA", "RENT3.SA", "LREN3.SA", "MGLU3.SA", 
    "POMO4.SA", "MRFG3.SA", "BEEF3.SA", "MOTV3.SA", "MRVE3.SA", 
    "MULT3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "RECV3.SA", 
    "PRIO3.SA", "PETZ3.SA", "PSSA3.SA", "RADL3.SA", "RAIZ4.SA", 
    "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "SANB11.SA", "STBP3.SA", 
    "SMTO3.SA", "CSNA3.SA", "SLCE3.SA", "SMFT3.SA", "SUZB3.SA", 
    "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", 
    "USIM5.SA", "VALE3.SA", "VAMO3.SA", "VBBR3.SA", "VIVA3.SA", 
    "WEGE3.SA", "YDUQ3.SA"
]


SP500_TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ABMD", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO", "AVB", "AVY", "BKR", "BALL", "BAC", "BBWI", "BAX", "BDX", "WRB", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW", "CDNS", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CF", 
    "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMCSA", "CMA", "CAG", "COP", "ED", "STZ", "CPRT", "GLW", "CTVA", "COST", "CTRA", "CCI", "CSX", "CMI", "CVS", "DHI", "DHR", "DRI", "DVA", "DE", "DAL", "XRAY", "DVN", "DXCM", "FANG", "DLR", "DFS", "DIS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", "DTE", "DUK", "DD", "DXC", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FAST", "FRT", "FDX", "FITB", "FRC", "FE", "FIS", "FISV", "FLT", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GS", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", 
    "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "ILMN", "INCY", "IR", "INTC", "ICE", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR", "K", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LNC", "LIN", "LYV", "LKQ", "LMT", "L", "LOW", "LUMN", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", "NWL", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE", "ORCL", "OGN", "OTIS", "PCAR", "PKG", "PARA", "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PTC", 
    "PSA", "PHM", "PVH", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "RHI", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SEDG", "SO", "LUV", "SWK", "SBUX", "STT", "STE", "SYK", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VFC", "VTRS", "VICI", "V", "VMC", "WAB", "WMT", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WRK", "WY", "WHR", "WMB", "WTW", "GWW", "WYNN", "XEL", "XYL", "YUM",    "ZBRA", "ZBH", "ZION", "ZTS"
]


# =============================================
# INTERFACE PRINCIPAL
# =============================================

def main():
    st.set_page_config(page_title="Analisador Graham & Greenblatt", layout="wide")
    st.title("📊 Analisador de Ações - Graham & Greenblatt")
    
    # Sidebar
    with st.sidebar:
        st.header("Configurações")
        indice = st.selectbox("Escolha o índice:", ["IBOVESPA", "S&P 500"])
        
        st.subheader("Pesquisa Rápida")
        ticker_direto = st.text_input("Digite um ticker para ver detalhes:")
        if ticker_direto and st.button("Buscar"):
            if not (ticker_direto.endswith('.SA') or '.' in ticker_direto):
                if indice == "IBOVESPA":
                    ticker_direto = f"{ticker_direto}.SA"
            mostrar_detalhes_empresa(ticker_direto)
            st.stop()
    
    # Carregar dados
    with st.spinner("Carregando dados... Isso pode levar alguns segundos."):
        start_time = time.time()
        df = get_stock_data(IBOV_TICKERS if indice == "IBOVESPA" else SP500_TICKERS)
        st.success(f"Dados carregados em {time.time() - start_time:.2f} segundos!")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    with col1:
        filtro_graham = st.selectbox("Filtro Graham", ["Todos", "Barata", "Cara", "Indefinido"])
    with col2:
        filtro_magic = st.selectbox("Filtro Magic Formula", ["Todos", "Barata", "Cara", "Indefinido"])
    with col3:
        ordenar_por = st.selectbox("Ordenar por", ["Ticker", "Preço", "P/L", "P/VP", "ROE"])
    
    # Aplicar filtros e ordenação
    filtered_df = df.copy()
    if filtro_graham != "Todos":
        filtered_df = filtered_df[filtered_df["Graham"] == filtro_graham]
    if filtro_magic != "Todos":
        filtered_df = filtered_df[filtered_df["Magic"] == filtro_magic]
    
    if ordenar_por != "Ticker":
        filtered_df = filtered_df.sort_values(by=ordenar_por, ascending=False)
    
    # Mostrar resultados
    st.dataframe(
        filtered_df.reset_index(drop=True).style.apply(destacar_linhas, axis=1),
        use_container_width=True
    )
    
    # Estatísticas
    st.subheader("Estatísticas da Análise")
    cols = st.columns(3)
    cols[0].metric("Total de Ações", len(df))
    cols[1].metric("Baratas (Graham)", len(df[df["Graham"] == "Barata"]))
    cols[2].metric("Baratas (Magic Formula)", len(df[df["Magic"] == "Barata"]))
    
    # Detalhes da ação selecionada
    st.subheader("Selecione uma ação para ver detalhes")
    selected_ticker = st.selectbox("Escolha um ticker:", [""] + filtered_df["Ticker"].tolist())
    if selected_ticker:
        mostrar_detalhes_empresa(selected_ticker)
    
    # Rodapé
    st.caption("Fonte dos dados: Yahoo Finance via yfinance. Análise educacional.")
    st.caption("Disclaimer: A análise apresentada é apenas para fins educativos.")

if __name__ == "__main__":
    main()
