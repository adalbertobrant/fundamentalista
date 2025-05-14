# 📈 Analisador de Ações - IBOVESPA & S&P 500

Este é um aplicativo Streamlit que permite análise fundamentalista e técnica de ações das bolsas brasileira (IBOVESPA) e americana (S&P 500). A aplicação faz uso dos métodos de Benjamin Graham e Joel Greenblatt para classificar ações como "baratas" ou "caras", além de exibir gráficos interativos e indicadores técnicos como RSI e MACD.

## 🚀 Funcionalidades

- 🔎 Busca e análise de ações por Ticker
- 📊 Indicadores Fundamentalistas:
  - P/L (Preço/Lucro)
  - P/VP (Preço/Valor Patrimonial)
  - ROE (Retorno sobre Patrimônio)
- 📈 Classificação por:
  - 📘 Método Graham
  - 🧙‍♂️ Fórmula Mágica de Greenblatt
- 📉 Análise Técnica:
  - RSI (Índice de Força Relativa)
  - MACD (Média Móvel Convergente e Divergente)
- 🏢 Exibição de informações completas da empresa:
  - Setor, indústria, site, executivos, resumo do negócio
- 📦 Processamento paralelo e cache inteligente
- 🇧🇷 Ações do IBOVESPA e 🇺🇸 S&P 500

## 🛠️ Tecnologias Utilizadas

- [Streamlit](https://streamlit.io)
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)
- [functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)

## 🧠 Classificação de Valor

### 📘 Benjamin Graham
Classifica uma ação como **Barata** se:

​		P/L * P/VP < 22.5

### 🧙‍♂️ Joel Greenblatt (Fórmula Mágica)
Classifica uma ação como **Barata** se:

​		P/L < 15 e ROE > 15%



## 📉 Indicadores Técnicos

- **RSI (Relative Strength Index):** Mede a força do movimento de preço (sobrecompra/sobrevenda).
- **MACD (Moving Average Convergence Divergence):** Detecta tendências e mudanças de momentum.

## ⚙️ Execução

0. Crie um ambiente virutal
   ````
   python3 -m venv ambientevirtual

   source ambientevirtual/bin/activate

   ````

1. Clone o repositório:

````
git clone (https://github.com/adalbertobrant/fundamentalista.git)
cd fundamentalista



Instale os requisitos:

````
pip install -r requirements.txt
```
Execute o app
```
streamlit run app.py
```

📁 Estrutura
app.py: Código principal da aplicação

README.md: Documentação do projeto

requirements.txt: Dependências do projeto

✅ Requisitos
Python 3.8+

Bibliotecas:

streamlit

yfinance

pandas

numpy

matplotlib

pillow

requests

🧠 Ideias Futuras
Filtros avançados por setor, indústria e dividendos

Exportação de resultados para CSV ou Excel

Integração com APIs alternativas como Finnhub ou Alpha Vantage

Comparativo entre empresas

👨‍💻 Se achou legal mande umas stars não custa nada e me ajuda em um estágio!!!!
