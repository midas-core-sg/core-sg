# Guia Rápido - Executar Benchmarks de Qualidade CoreSG

## TL;DR (3 minutos)

### Na sua máquina atual (para criar os scripts):
✅ **Já feito** - Scripts criados em:
- `benchmarking/quality/run_quality_comparison_to_50k.sh` (recomendado)
- `benchmarking/quality/run_article_quality_to_50k.sh` (avançado)
- `benchmarking/quality/SETUP_AND_RUN.md` (guia completo)

---

## Em uma máquina nova

### 1️⃣ Setup Inicial (10 min)

```bash
# Clone o repositório
git clone <url> core-sg
cd core-sg

# Crie ambiente Python
python3.10 -m venv .venv310
source .venv310/bin/activate

# Instale CoreSG
pip install -e .

# Teste
python -c "from core_sg import CoreSG; print('✓ OK')"
```

### 2️⃣ Execute os Benchmarks (2-8 horas)

**Opção A - Simplificado (recomendado):**
```bash
source .venv310/bin/activate
THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Opção B - Em background (máquinas remotas):**
```bash
nohup bash -c 'source .venv310/bin/activate && THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh' > benchmark.log 2>&1 &
tail -f benchmark.log
```

**Opção C - Configuração Customizada:**
```bash
source .venv310/bin/activate
THREADS=8 SAMPLES_CSV="10000,50000" DIMENSIONS_CSV="20" ./benchmarking/quality/run_article_quality_to_50k.sh
```

### 3️⃣ Verifique Resultados

```bash
# Arquivos principais
ls -lh benchmarking/quality/results/quality_comparison*.csv

# Amostra de resultados
head -5 benchmarking/quality/results/quality_comparison_by_k.csv
```

---

## Detalhes dos Scripts

### Script 1: `run_quality_comparison_to_50k.sh` ⭐

**Quando usar:** Experimentos padrão do artigo

**Features:**
- ✓ Configuração pré-definida fixa
- ✓ Todas as 9 distribuições
- ✓ Todas as 6 dimensões (2, 10, 20, 32, 64, 128)
- ✓ Todos os 6 tamanhos de amostra (5k-50k)
- ✓ 4 métodos

**Uso básico:**
```bash
THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Com argumentos extras:**
```bash
./benchmarking/quality/run_quality_comparison_to_50k.sh --limit 10
SKIP_EXISTING=1 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

### Script 2: `run_article_quality_to_50k.sh`

**Quando usar:** Experimentos customizados

**Variáveis de ambiente:**
```bash
SAMPLES_CSV="5000,10000,20000"          # Tamanhos de amostra (máx 50k)
DIMENSIONS_CSV="2,10,20,32,64,128"      # Dimensões
DISTRIBUTIONS_CSV="gaussian,poisson"    # Distribuições (9 disponíveis)
THREADS=4                                # Threads para usar
K_MAX=50                                 # Parâmetro k máximo
SKIP_EXISTING=1                          # Retomar run interrompida
RUN_HDBSCAN_FAMILY=1                    # Rodar HDBSCAN methods
RUN_SCORESG_FAMILY=1                    # Rodar ScoreSG methods
```

**Exemplo:**
```bash
export SAMPLES_CSV="5000,10000,50000"
export DIMENSIONS_CSV="20"
export DISTRIBUTIONS_CSV="gaussian,gaussian_sparse"
export THREADS=8
export SKIP_EXISTING=1
./benchmarking/quality/run_article_quality_to_50k.sh
```

---

## Distribuições Disponíveis

As 9 distribuições sintéticas testadas:
1. `gaussian` - Distribuição normal
2. `poisson` - Distribuição de Poisson
3. `chi_square` - Distribuição qui-quadrado
4. `gamma` - Distribuição gama
5. `beta` - Distribuição beta
6. `von_mises` - Distribuição von Mises
7. `gumbel` - Distribuição Gumbel
8. `logistic` - Distribuição logística
9. `gaussian_sparse` - Normal com esparsidade

---

## Métodos Comparados

1. **HDBSCAN (Referência Exata)** - `hdbscan_generic`
2. **HDBSCAN Otimizado** - `optimized_hdbscan`
3. **ScoreSG** - `score_sg`
4. **ScoreSG Aleatório** - `score_sg_random`

---

## Métricas de Saída

| Métrica | Plano | Descrição |
|---------|-------|-----------|
| `ari_vs_hdbscan_generic` | -1 a 1 | Quanto cada método concorda com HDBSCAN exato |
| `hai` | -1 a 1 | Concordância da hierarquia de clustering |
| `mst_edge_jaccard` | 0 a 1 | Sobreposição de arestas MST |

---

## Otimizações de Performance

### Para máquinas potentes (16+ cores):
```bash
THREADS=12 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

### Para teste rápido (5 min):
```bash
SAMPLES_CSV="5000" DIMENSIONS_CSV="2" DISTRIBUTIONS_CSV="gaussian" \
  PYTHON_BIN=".venv310/bin/python" ./benchmarking/quality/run_article_quality_to_50k.sh
```

### Para produção (máquina remota):
```bash
nohup bash -c '
  cd /path/to/core-sg
  source .venv310/bin/activate
  THREADS=8 SKIP_EXISTING=1 ./benchmarking/quality/run_quality_comparison_to_50k.sh
' > quality_benchmark.log 2>&1 &
```

---

## Monitoramento

```bash
# Acompanhar progresso (em outra janela)
tail -f benchmark.log

# Procurar erros
grep -i error benchmark.log

# Contar configurações completadas
tail -100 benchmark.log | grep "Configuration completed"
```

---

## Retomar Execução Interrompida

```bash
SKIP_EXISTING=1 THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

---

## Problemas Comuns

### ❌ "ModuleNotFoundError: No module named 'core_sg'"
```bash
cd core-sg
pip install -e .
python -c "from core_sg import CoreSG; print('OK')"
```

### ❌ Memory errors
```bash
# Reduza tamanho
SAMPLES_CSV="5000,10000" DIMENSIONS_CSV="2,10" THREADS=1 ./run_...sh
```

### ❌ Excede 50k samples
O script valida automaticamente. Máximo permitido: 50000 (por estar usando HDBSCAN exato)

---

## Saída Esperada

```
benchmarking/quality/results/
├── quality_comparison_by_k.csv          ← Resultados detalhados
├── quality_comparison_summary.csv       ← Resumo por método/dataset
├── quality_comparison_manifest.json     ← Metadados
└── _runs/                               ← Outputs individuais por config
```

---

## Próximos Passos

1. **Para instruções completas:** Leia `SETUP_AND_RUN.md`
2. **Para executar na máquina nova:** Siga a seção "Em uma máquina nova" acima
3. **Para análise de resultados:** Verifique os arquivos CSV em `results/`

---

## Tempo Estimado

| Configuração | 1 CPU | 4 CPUs | 8 CPUs |
|---|---|---|---|
| 1 distrib, 6 dims, 6 sizes | 30 min | 10 min | 6 min |
| 9 distribs, 6 dims, 6 sizes | 4.5 hrs | 1.5 hrs | 1 hr |

⏱️ Para o setup completo: **2-8 horas** (depende de máquina e threads)
