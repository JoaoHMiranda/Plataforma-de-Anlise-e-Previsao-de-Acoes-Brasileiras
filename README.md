# Plataforma de Análise & Previsão de Ações

Aplicação para **análise exploratória**, **treino de modelos** (SVR/LSTM com vários otimizadores) e **previsão** de preços de ações — diretamente no navegador.

> **Aviso**: uso **exclusivamente educacional**. Os resultados são estimativas e **não constituem recomendação de investimento**.

---

## ✨ Destaques

* **Single file**: tudo no `app.py`. Modelos salvos em `./modelos/` e datasets em `./datasets/`.
* **Coleta de dados com fallback**: Yahoo Finance → BRAPI → Stooq (ordem configurável, modo “insistente” opcional).
* **Análise gráfica**:

  * Candlestick + MM20/MM50
  * RSI(14), boxplot do fechamento, retornos e barras de volume anual
  * KPIs (último preço, retorno 21d, vol anualizada \~21d, máx 52s)
  * Download de um **.zip** com TXT (resumo), CSV e PNGs
* **Treino de modelos**:

  * **SVR** (RBF) e **LSTM**
  * Modos: **Manual**, **Bayes**, **Grid**, **Genético (GA)**, **PSO**
  * A **UI muda automaticamente** conforme o modo (exibe apenas os parâmetros relevantes)
  * Robustificação anti-erros de forma: criação de janelas, split time-series e **CV segura** (evita “folds > samples”)
  * Fallback interno rápido (mini-grid SVR) em caso de falha
* **Salvamento/gerenciamento de modelos**:

  * **SVR**: `*.svr.pkl` (inclui `SVR`, `MinMaxScaler` e metadados)
  * **LSTM**: `*.keras` + `*.pkl` (escala + metadados)
  * `*.meta.json` com descrição/ticker/período/lookback etc.
* **Previsão**:

  * Carrega um modelo salvo e prevê *n* dias úteis
  * Gráfico (histórico + previsão) e **download do CSV**
* **GPU opcional**: se existir, usa com *memory growth*; caso contrário, força CPU silenciosa.

---

## 📦 Instalação

Recomendado Python 3.10+.

```bash

# Ambiente (opcional mas recomendado)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependências
pip install -U pip
pip install streamlit yfinance plotly scikit-learn pandas numpy matplotlib joblib requests tensorflow

# Otimizadores (opcional, por modo)
pip install scikit-optimize      # Bayes (SVR e LSTM)
pip install geneticalgorithm     # Genético (SVR)
pip install pyswarms             # PSO (SVR)
```

> Dica: se preferir, crie um `requirements.txt` com as libs acima.

---

## ▶️ Como rodar

```bash
streamlit run app.py
```

Acesse o link que o Streamlit mostrar (geralmente `http://localhost:8501`).

---

## 🧭 Guia rápido de uso

### 1) Aba **📊 Análise de Ações**

* Informe **ticker** (ex.: `PETR4.SA`) e **período** (ex.: `5y`, `6mo`, `1y`…).
* Escolha a **ordem das fontes** e, se quiser, ative o **modo insistente**.
* Clique **Analisar na Tela**:

  * KPIs, candlestick, MM20/MM50, RSI, boxplot, retornos e volume por ano.
  * **Salvar (Downloads)**: gera `.zip` com resumo TXT, CSV e imagens PNG.

### 2) Aba **🔬 Treinar Modelos (SVR/LSTM)**

* Selecione **modelo** (SVR ou LSTM) e **modo** (Manual / Bayes / Grid / Genético / PSO).
* A interface **adapta dinamicamente** os campos:

  * **SVR Manual**: `look_back`, `C`, `gamma`, `epsilon`
  * **SVR Bayes**: `look_back`, **Iterações (Bayes)** (min **10**), busca em (C, γ, ε)
  * **SVR Grid**: listas de `C`, `gamma`, `epsilon` + `look_back`
  * **SVR GA**: `População`, `Iterações`, `Prob. mutação/crossover`, `elit_ratio`, `parents_portion` + `look_back`
  * **SVR PSO**: `Partículas`, `Iterações`, `c1`, `c2`, `w`, **bounds (C,γ,ε)** + `look_back`
  * **LSTM Manual**: `look_back`, `units`, `dropout`, `recurrent_dropout`, `epochs`, `batch_size`
  * **LSTM Bayes**: faixas de `units`, `dropout`, `epochs` + `n_calls` (min **10**) + `look_back`
  * **LSTM GA (simplificado)**: `População`, `Gerações` + `look_back`
  * **LSTM PSO (simplificado)**: `Tentativas` (busca aleatória guiada) + `look_back`
* Clique **Treinar**:

  * Exibe métrica (RMSE/MAE/R²), gráfico “Real vs Previsto” e, no caso de LSTM, curva de loss.
  * Modelos salvos automaticamente em `./modelos/`.
  * Botão **Salvar pacote do treino (apenas modelos)** gera `.zip` só com os artefatos de modelo.

### 3) Aba **📈 Fazer Previsão**

* Selecione um modelo salvo.
* Informe **ticker** (pode ser outro), **horizonte (dias úteis)** e **período** para carregar a série.
* Clique **Rodar Previsão** para ver o gráfico e **baixar CSV** dos valores previstos.

---

## 🔧 Detalhes técnicos & decisões

* **Criação de janelas**: séries “Close” normalizadas com `MinMaxScaler` e janelas `look_back`.
* **Robustificação contra erros comuns**:

  * Evita arrays 1D/vazios no split treino/teste.
  * **TimeSeriesSplit** “seguro”: escolhe `n_splits` válido; se impraticável, cai para *hold-out*.
  * Se a série for muito curta, gera **dados sintéticos suaves** para fins didáticos.
* **Bayes (skopt)**:

  * **SVR** usa `BayesSearchCV` (C, γ, ε).
  * **LSTM** usa `gp_minimize` para `units`, `dropout`, `epochs`, `batch`.
  * **n\_calls / Iterações ≥ 10** (UI e backend).
* **PSO (pyswarms)** para **SVR**: partículas, iterações e hiperparâmetros clássicos (`c1`, `c2`, `w`), com **bounds** para (C, γ, ε).
  **Obs.**: o campo `look_back` é explicitamente solicitado no PSO (fix para NameError).
* **Genético (geneticalgorithm)** para **SVR**: busca real contínua em (C, γ, ε) com configuração de população, iterações, elitismo, etc.
* **LSTM GA/PSO**: versões leves (heurísticas simplificadas) para manter tudo em um arquivo.

---

## 🗂 Estrutura gerada

```
.
├─ app.py                # único arquivo da aplicação
├─ modelos/              # modelos salvos + metadados
│  ├─ <nome>_YYYYmmdd-HHMMSS.svr.pkl
│  ├─ <nome>_YYYYmmdd-HHMMSS.keras
│  ├─ <nome>_YYYYmmdd-HHMMSS.pkl        # scaler + params (LSTM)
│  └─ <nome>_YYYYmmdd-HHMMSS.meta.json  # informações do modelo
└─ datasets/
   └─ dados_<TICKER>_<PERIODO>_processado.csv
```

---

## 🔌 Fontes de dados e tickers

* **Yahoo Finance**, **BRAPI** e **Stooq** (com ordem configurável e modo insistente).
* Para ações brasileiras, use sufixo `.SA` (ex.: `PETR4.SA`, `VALE3.SA`).

---

## 🧪 Solução de problemas

* **“Expected 2D array, got 1D array instead: array=\[] …”**
  Resolvido no código com criação de janelas, *reshape* e splits seguros. Se persistir, aumente o período ou `look_back` menor.
* **“Cannot have number of folds=… greater than number of samples=…”**
  O app reduz automaticamente os `n_splits` e, se necessário, cai para *hold-out*. Experimente ampliar o período.
* **Sem dados do provedor**
  Ative **modo insistente**, troque a **ordem das fontes** e verifique o **ticker**. Como demonstração, o app pode gerar série sintética.
* **Limites de API (rate limit)**
  Tente novamente após alguns minutos; a app já silencia logs e tem cache leve (TTL 900s para Yahoo).

---

## 🔐 Aviso legal

Este projeto tem **finalidade acadêmica/educacional**.
Nenhuma previsão deve ser entendida como orientação financeira.

---

## 🙌 Créditos

Desenvolvido por **João Henrique Silva de Miranda**, com apoio do **CNPq** e da **PUC Goiás**.
Links no rodapé do app (LinkedIn/GitHub).

---

### 📜 Licença

Defina a licença do repositório MIT.
