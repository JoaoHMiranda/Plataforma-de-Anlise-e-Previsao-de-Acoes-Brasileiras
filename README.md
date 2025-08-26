# Plataforma de Análise e Previsão de Ações

Aplicação web em **Streamlit** para:

* **Analisar** séries históricas de ações (candlestick, médias móveis, RSI, KPIs, etc.);
* **Treinar modelos de IA** (☑️ **SVR** e ☑️ **LSTM**) com ou sem otimizadores (Bayes, Grid, Genético, PSO);
* **Executar previsões** com modelos salvos e baixar os resultados.

> ⚠️ **Uso exclusivamente educacional.** Os resultados são estimativas e **não constituem recomendação de investimento**.

---

## ✨ Funcionalidades

* **Fontes de dados com fallback**: Yahoo → BRAPI → Stooq (ordem configurável).
* **Indicadores**: MM20, MM50, RSI(14), retornos, volatilidade anualizada.
* **Treino de modelos**

  * **SVR**: manual / Bayes / Grid / Genético / PSO.
  * **LSTM**: manual / Bayes / Genético / PSO.
  * **Barra de progresso** acompanha o treino em tempo real (via stdout + tempo).
  * **GPU quando disponível**; se não houver, roda automaticamente em **CPU**.
* **Persistência de modelos**

  * Tudo salvo em `./modelos/` (nome: `<DisplayName>_<timestamp>.(keras|pkl|svr.pkl)`).
  * **Imagens de treino não ficam salvas** — são exibidas na página e descartadas.
* **Previsão**

  * Selecione qualquer modelo salvo, escolha horizonte e período, visualize e **baixe CSV**.
* **Exportação**

  * Aba de análise permite baixar um `.zip` com **TXT de resumo**, **CSV** e **gráficos** do histórico.
  * Aba de treino gera um `.zip` com **apenas os modelos** (sem imagens/TXT de treino).

---

## 🗂️ Estrutura (resumo)

```
.
├── app.py
├── modelos/                 # pasta onde os modelos salvos são consolidados
│   └── <Display>_<ts>.keras / .pkl / .svr.pkl / .meta.json
├── datasets/                # CSVs temporários para scripts externos
├── (opcional) scripts de otimização:
│   ├── svr_bayes.py         ├── svr_grid.py
│   ├── svr_genetico.py      ├── svr_pso.py
│   ├── lstm_bayes.py        ├── lstm_genetico.py
│   └── lstm_pso.py
└── README.md
```

> Os scripts de otimização podem ficar em `./modelos` (padrão) ou em outra pasta que você apontar na interface.

---

## ⚙️ Requisitos

* **Python 3.10 – 3.12**
* (Opcional) **GPU NVIDIA** com drivers/CUDA compatíveis se quiser acelerar LSTM.

---

## 🚀 Instalação rápida

### 1) Crie um ambiente virtual

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Instale as dependências

**Opção A – tudo via `pip` diretamente**

```bash
pip install streamlit plotly numpy pandas scikit-learn tensorflow yfinance requests joblib matplotlib scikit-optimize geneticalgorithm pyswarms
```

> 💡 **GPU (opcional)**: em muitas instalações recentes você pode usar
> `pip install "tensorflow[and-cuda]"`
> Caso não tenha GPU ou queira evitar CUDA, `pip install tensorflow` (CPU) já funciona.

**Opção B – usando um `requirements.txt` (opcional)**
Crie um arquivo `requirements.txt` com o conteúdo abaixo e rode `pip install -r requirements.txt`:

```
streamlit
plotly
numpy
pandas
scikit-learn
tensorflow
yfinance
requests
joblib
matplotlib
scikit-optimize
geneticalgorithm
pyswarms
```

---

## ▶️ Como rodar

```bash
streamlit run app.py
```

* Acesse o link local exibido no terminal (ex.: `http://localhost:8501`).
* Se houver GPU disponível, a aplicação tenta usá-la automaticamente; caso contrário, segue em CPU.

---

## 🧭 Como usar

1. **📊 Análise de Ações**

   * Digite o **ticker** (ex.: `PETR4.SA`), escolha o período e clique em **Analisar na Tela**.
   * Veja candlestick, MMs, RSI, retornos e métricas.
   * Baixe o pacote `.zip` (TXT + CSV + gráficos do **histórico**).

2. **🔬 Treinar Modelos (SVR/LSTM)**

   * Escolha **modelo** e **otimizador**.
   * Para **manual**, ajuste hiperparâmetros e treine.
   * Para **otimizadores externos** (Bayes, Grid, Genético, PSO), aponte a pasta onde estão os scripts
     (`svr_bayes.py`, `lstm_pso.py`, etc.).
   * Ao final, os modelos são **consolidados em `./modelos/`** com o nome que você definiu.
   * As **imagens** aparecem no site e depois são **descartadas**; o ZIP de treino contém **apenas os modelos**.

3. **📈 Fazer Previsão**

   * Selecione um modelo salvo, defina o horizonte (dias úteis) e rode.
   * Baixe o **CSV** com a curva prevista.

---

## 🧪 Otimizadores externos (extras)

* **SVR**: `svr_bayes.py`, `svr_grid.py`, `svr_genetico.py`, `svr_pso.py`
* **LSTM**: `lstm_bayes.py`, `lstm_genetico.py`, `lstm_pso.py`

Dependências adicionais já estão na lista (✅ `scikit-optimize`, ✅ `geneticalgorithm`, ✅ `pyswarms`).
Coloque os scripts em `./modelos` (padrão) **ou** ajuste o caminho na interface antes de treinar.

---

## 🛟 Dicas & Solução de problemas

* **Mensagens CUDA/cuDNN no terminal**
  Se não houver GPU disponível, o TensorFlow roda **em CPU**. Essas mensagens podem ser ignoradas.
* **Limites de API/Yahoo/BRAPI**
  Se faltar dado, a aplicação tenta novas fontes e pode insistir automaticamente (opção “modo insistente”).
* **Dependência faltando**
  O app avisa qual pacote instalar (ex.: `pip install pyswarms`).
* **Barra de progresso**
  A barra vai do tempo estimado + “saltos” quando encontra mensagens do treino (`Treinando…`, `Avaliando…`, `finalizados`, etc.).

---

## 👤 Autor

**João Henrique Silva de Miranda**
LinkedIn: [www.linkedin.com/in/joao-henrique-silva-de-miranda](https://www.linkedin.com/in/joao-henrique-silva-de-miranda)

---

## 🙏 Agradecimentos

Projeto desenvolvido com apoio e financiamento do **Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq)** e da **Pontifícia Universidade Católica de Goiás (PUC Goiás)**. Muito obrigado! 🎓

---

## 📜 Licença

Defina aqui a licença do projeto MIT.
