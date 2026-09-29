# ✅ Arquivos Criados - Resumo Executivo

## 📋 Arquivos Gerados

### 1. **run_quality_comparison_to_50k.sh** ⭐ (RECOMENDADO)
**Arquivo:** `benchmarking/quality/run_quality_comparison_to_50k.sh`

**O que faz:**
- Script simples e direto para rodar os experimentos de qualidade conforme o artigo
- Configuração fixa: todas as 9 distribuições, todos os 6 tamanhos de amostra, todas as 6 dimensões
- 4 métodos: HDBSCAN, Optimized HDBSCAN, ScoreSG, ScoreSG Random
- Métricas: ARI (Adjusted Rand Index) vs. HDBSCAN e HAI (Hierarchy Agreement Index)

**Como usar:**
```bash
cd core-sg
source .venv310/bin/activate
THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**Variáveis de ambiente suportadas:**
- `THREADS` - Número de threads (padrão: 1)
- `PYTHON_BIN` - Caminho do Python (padrão: python3)
- `OUTPUT_DIR` - Diretório de resultados (padrão: benchmarking/quality/results)
- `SKIP_EXISTING` - Retomar run interrompida (padrão: 0)
- `LOG_LEVEL` - Nível de log (padrão: INFO)

---

### 2. **run_article_quality_to_50k.sh** (AVANÇADO)
**Arquivo:** `benchmarking/quality/run_article_quality_to_50k.sh`

**O que faz:**
- Script completo com total controle sobre configurações
- Suporta todas as variáveis de ambiente do script simplificado + mais
- Loop estruturado sobre todas as configurações
- Melhor para experimentos customizados e reproducibilidade

**Variáveis suportadas:**
- `SAMPLES_CSV` - Tamanhos de amostra (padrão: 5000,10000,20000,30000,40000,50000)
- `DIMENSIONS_CSV` - Dimensões (padrão: 2,10,20,32,64,128)
- `DISTRIBUTIONS_CSV` - Distribuições (padrão: todas as 9)
- `THREADS` - Threads (padrão: 1)
- `K_MIN`, `K_MAX` - Parâmetros de clustering
- `RUN_HDBSCAN_FAMILY`, `RUN_SCORESG_FAMILY` - Quais famílias de métodos rodar
- `SKIP_EXISTING` - Retomar run
- Muitas outras (veja o script para detalhes)

**Exemplo de uso customizado:**
```bash
export SAMPLES_CSV="5000,10000,50000"
export DIMENSIONS_CSV="20,64"
export DISTRIBUTIONS_CSV="gaussian,gaussian_sparse"
export THREADS=8
./benchmarking/quality/run_article_quality_to_50k.sh
```

---

### 3. **SETUP_AND_RUN.md** (GUIA COMPLETO)
**Arquivo:** `benchmarking/quality/SETUP_AND_RUN.md`

**O que contém:**
1. **Part 1: Prerequisites** - Requisitos de sistema e software
2. **Part 2: Repository Setup** - Como clonar, make venv, instalar CoreSG
3. **Part 3: Configuration Options** - Detalhe comparativo dos dois scripts
4. **Part 4: Running Experiments** - Exemplos de execução
5. **Part 5: Understanding Outputs** - Estrutura e significado dos resultados
6. **Part 6: Performance & Optimization** - Dicas de performance
7. **Part 7: Troubleshooting** - Solução de problemas comuns
8. **Part 8: Validation** - Como validar que os resultados estão corretos
9. **Part 9: Advanced Usage** - Configurações personalizadas, paralelização
10. **Part 10: Checklist** - Checklist final de verificação

**Público-alvo:** Qualquer pessoa configurando em uma máquina nova

---

### 4. **QUICKSTART.md** (GUIA RÁPIDO)
**Arquivo:** `benchmarking/quality/QUICKSTART.md`

**O que contém:**
- Instruções de 3 minutos (TL;DR)
- Setup inicial passo-a-passo
- 3 opções de execução (simples, background, customizado)
- Verificação de resultados
- Guia de distribuições disponíveis
- Tabela de performance estimada
- Troubleshooting rápido

**Público-alvo:** Usuários que querem sair do zero para rodando em < 15 min

---

## 📊 Resumo das Configurações

### Tamanhos de Amostra (N)
- 5.000
- 10.000
- 20.000
- 30.000
- 40.000
- 50.000
**Máximo:** 50.000 (limitação do HDBSCAN exato)

### Dimensões (D)
- 2 (muito baixa)
- 10 (baixa)
- 20 (média-baixa)
- 32 (média)
- 64 (média-alta)
- 128 (alta)

### Distribuições (9 total)
1. gaussian
2. poisson
3. chi_square
4. gamma
5. beta
6. von_mises
7. gumbel
8. logistic
9. gaussian_sparse

### Métodos (4 total)
1. HDBSCAN (referência exata) - `hdbscan_generic`
2. HDBSCAN Otimizado - `optimized_hdbscan`
3. ScoreSG - `score_sg`
4. ScoreSG Aleatório - `score_sg_random`

### Métricas Computadas
- **ARI (Adjusted Rand Index)** - Concordância com HDBSCAN exato (-1 a 1)
- **HAI (Hierarchy Agreement Index)** - Concordância da hierarquia de clustering (-1 a 1)
- **MST Edge Jaccard** - Sobreposição de arestas MST (0 a 1)

---

## ⏱️ Tempo de Execução Estimado

| Configuração | 1 CPU | 4 CPUs | 8 CPUs |
|---|---|---|---|
| 1 distribuição, 6 dimensões, 6 tamanhos | **30 min** | **10 min** | **6 min** |
| 9 distribuições, 6 dimensões, 6 tamanhos | **4.5 hrs** | **1.5 hrs** | **1 hr** |
| Com datasets reais adicionados | **6 hrs** | **2 hrs** | **1.5 hrs** |

**Fator limitante:** O cálculo exato de HAI é O(N²) em pares de pontos. Para N=50k, isso é ~1.25 bilhões de comparações por método.

---

## 🎯 Próximas Ações

### Para Rodar Imediatamente nesta Máquina

Você JÁ pode rodar um teste rápido:

```bash
cd /home/gab04/Desktop/core-sg

# Prepare environment (if not already done)
python3.10 -m venv .venv310
source .venv310/bin/activate
pip install -e .

# Test run (5 minutos)
SAMPLES_CSV="5000" DIMENSIONS_CSV="2" DISTRIBUTIONS_CSV="gaussian" \
  PYTHON_BIN=".venv310/bin/python" ./benchmarking/quality/run_article_quality_to_50k.sh

# Full run (2-8 horas)
THREADS=4 PYTHON_BIN=".venv310/bin/python" SKIP_EXISTING=1 \
  ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

### Para Rodar em Máquina Nova

1. **Leia o guia:** `SETUP_AND_RUN.md` (10-15 min)
2. **Execute setup:** Parte 2 + 3 (10 min)
3. **Execute benchmarks:** Parte 4 (2-8 horas)
4. **Verifique resultados:** Parte 5 (5 min)

Ou, para retomar rápido: Veja `QUICKSTART.md`

---

## 📁 Estrutura de Resultados

Após execução, você terá:

```
benchmarking/quality/results/
├── quality_comparison_by_k.csv          # Resultados detalhados (principal)
├── quality_comparison_summary.csv       # Resumo agregado por método/distribuição
├── quality_comparison_manifest.json     # Metadados & status
└── _runs/                               # Outputs de cada configuração
    ├── synthetic_gaussian_d2_n5000_seed42/
    ├── synthetic_gaussian_d2_n10000_seed42/
    ├── synthetic_poisson_d2_n5000_seed42/
    └── ... (um por combinação de dist/dim/size/seed)
```

**Arquivo Principal:** `quality_comparison_by_k.csv`
- Linhas: Combinação(método, distribuição, dimensão, tamanho, k, seed)
- Colunas: benchmark_group, dataset_name, method, i_mds, j_mds, ari_vs_hdbscan_generic, **hai**, mst_edge_jaccard, ...

---

## 🔧 Troubleshooting Rápido

| Problema | Solução |
|---|---|
| "ModuleNotFoundError: No module named 'core_sg'" | `pip install -e .` |
| Memory error | Reduza SAMPLES_CSV ou DIMENSIONS_CSV |
| "exceeds 50,000 limit" | Máximo é 50k (validação automática) |
| Execução muito lenta | Aumente THREADS, reduza DISTRIBUTIONS_CSV |
| Run interrompida | Use `SKIP_EXISTING=1` para retomar |

---

## 📝 Resumo Técnico

**Baseado em:** `run_article_runtime_to_50k.sh` (runtime) adaptado para qualidade

**Diferenças principais:**
- Runtime: Mede tempo de execução (segundos)
- Qualidade (novo): Mede qualidade de clustering (ARI, HAI)
- Runtime: Usa diversos tamanhos em loop rápido
- Qualidade: Avalida em profundidade com múltiplas métricas

**Métricas de Referência:**
- HDBSCAN exato é a baseline de qualidade (considerar essa a "verdade")
- ScoreSG aproximado deve ter ARI > 0.95 em geral
- HAI mede concordância hierárquica mais granular que ARI

---

## 📞 Dúvidas?

- **Setup/instalação:** Ver `SETUP_AND_RUN.md` Part 1-2
- **Como rodar:** Ver `QUICKSTART.md` ou `SETUP_AND_RUN.md` Part 4
- **Interpretar resultados:** Ver `SETUP_AND_RUN.md` Part 5
- **Problemas:** Ver `SETUP_AND_RUN.md` Part 7

---

## ✨ Status

✅ Scripts criados e executáveis  
✅ Documentação completa  
✅ Guias passo-a-passo  
✅ Pronto para usar em máquina nova  

🚀 **Próximo passo:** Transferir para máquina nova e executar!
