# 📚 Índice de Documentação - CoreSG Quality Benchmarks (ARI/HAI)

## 👋 Comece Aqui

Você tem um conjunto completo de scripts e documentação para rodar experimentos de **qualidade** (ARI e HAI) do CoreSG, limitado a **50k samples** conforme o artigo.

### Escolha o seu caminho:

---

## 🚀 1. TL;DR - Quer rodar em 3 minutos?

**Arquivo:** [QUICKSTART.md](QUICKSTART.md)

**Contém:**
- Setup em 10 minutos
- 3 formas de executar
- Troubleshooting rápido
- Tabela de tempos estimados

**[→ Ir para QUICKSTART.md](QUICKSTART.md)**

---

## 📖 2. Guia Completo - Passo-a-passo Detalhado

**Arquivo:** [SETUP_AND_RUN.md](SETUP_AND_RUN.md)

**10 partes cobrindo:**
1. Pré-requisitos de sistema
2. Setup do repositório
3. Opções de configuração
4. Exemplos de execução
5. Entender outputs
6. Otimização de performance
7. Troubleshooting
8. Validação de resultados
9. Uso avançado
10. Checklist final

**[→ Ir para SETUP_AND_RUN.md](SETUP_AND_RUN.md)**

---

## ✅ 3. Resumo Executivo - O que foi criado?

**Arquivo:** [CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md)

**Contém:**
- Lista dos 4 arquivos criados
- Explicação rápida de cada um
- Configurações padrão
- Tempo estimado de execução
- Troubleshooting rápido

**[→ Ir para CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md)**

---

## 🎬 Scripts Prontos para Usar

### Script 1: `run_quality_comparison_to_50k.sh` ⭐ RECOMENDADO

**Status:** ✅ Pronto - Executável

**Quando usar:** Experimentos padrão conforme artigo

**Características:**
- Configuração fixa e predefinida
- Todas as 9 distribuições
- Todos os 6 tamanhos (5k-50k)
- Todos os 6 dimensões
- 4 métodos (HDBSCAN + ScoreSG)

**Uso básico:**
```bash
source .venv310/bin/activate
THREADS=4 ./benchmarking/quality/run_quality_comparison_to_50k.sh
```

**[→ Ver detalhes em QUICKSTART.md](QUICKSTART.md)**

---

### Script 2: `run_article_quality_to_50k.sh` 

**Status:** ✅ Pronto - Executável

**Quando usar:** Experimentos customizados com total controle

**Características:**
- Variáveis de ambiente para tudo
- Loop estruturado com lógica clara
- Melhor para configurações específicas

**Exemplo customizado:**
```bash
export SAMPLES_CSV="10000,50000"
export DIMENSIONS_CSV="20,64"
export DISTRIBUTIONS_CSV="gaussian,gaussian_sparse"
./benchmarking/quality/run_article_quality_to_50k.sh
```

**[→ Ver detalhes em CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md)**

---

## 🎓 Entender a Estrutura

### Distribuições Disponíveis (9)
- gaussian
- poisson  
- chi_square
- gamma
- beta
- von_mises
- gumbel
- logistic
- gaussian_sparse

### Métodos Comparados (4)
1. HDBSCAN (referência exata)
2. HDBSCAN Otimizado
3. ScoreSG
4. ScoreSG Aleatório

### Métricas Computadas
- **ARI** - Adjusted Rand Index (-1 a 1)
- **HAI** - Hierarchy Agreement Index (-1 a 1)
- **MST Jaccard** - MST edge overlap (0 a 1)

### Tamanhos de Amostra
5000, 10000, 20000, 30000, 40000, 50000 (máx 50k)

### Dimensões
2, 10, 20, 32, 64, 128

---

## 🔄 Fluxo Recomendado

### Opção A: Setup + Execução Rápida (20 min + 1-8 hrs)

1. Ler [QUICKSTART.md](QUICKSTART.md) (3 min)
2. Seguir seção "Em uma máquina nova" (10 min de setup)
3. Executar com `run_quality_comparison_to_50k.sh` (2-8 hrs)
4. ✅ Resultados em `benchmarking/quality/results/`

### Opção B: Setup Detalhado + Execução (40 min + 1-8 hrs)

1. Ler [SETUP_AND_RUN.md](SETUP_AND_RUN.md) completo (20 min)
2. Seguir Partes 2-3 (20 min de setup)
3. Escolher opção em Parte 4
4. ✅ Resultados

### Opção C: Só Entender (sem executar ainda) (15 min)

1. Ler [CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md)
2. Ler [QUICKSTART.md](QUICKSTART.md)
3. ✅ Pronto para instruir alguém

---

## 📊 Resultados Esperados

Após execução, você terá 3 arquivos principais em `benchmarking/quality/results/`:

1. **quality_comparison_by_k.csv** ← Principal
   - Linhas completas para cada combinação de (distrib, dim, size, k, método)
   - Colunas: benchmark_group, dataset_name, method, ari_vs_hdbscan_generic, **hai**, ...

2. **quality_comparison_summary.csv**
   - Resumo agregado por método/distribuição

3. **quality_comparison_manifest.json**
   - Metadados da execução

---

## ⏱️ Tempos Estimados

| Tarefa | Tempo |
|--------|-------|
| Setup em máquina nova | 10-15 min |
| Teste rápido (1 dist, 1 dim) | 5 min |
| 1 distribuição completa | 30 min |
| Todas 9 distribuições (4 CPUs) | 1.5 - 2 hrs |
| Full benchmark (8 CPUs) | 1 - 1.5 hrs |

---

## 🎯 Mapa de Documentação

```
┌─ AQUI (INDEX.md) ─────────────────────────┐
│                                            │
│  ├─ QUICKSTART.md ─────────────────────┐ │
│  │ (3 min, essencial ler)              │ │
│  │                                      │ │
│  ├─ CREATED_FILES_SUMMARY.md ─────────┐ │
│  │ (5 min, saber o que foi criado)     │ │
│  │                                      │ │
│  ├─ SETUP_AND_RUN.md ──────────────────┐ │
│  │ (20 min, referência completa)       │ │
│  │                                      │
│  ├─ run_quality_comparison_to_50k.sh  │
│  │ (USAR ESTE - simples)               │
│  │                                      │
│  └─ run_article_quality_to_50k.sh     │
│    (Usar este - customização total)   │
└──────────────────────────────────────────┘
```

---

## 🚨 Verificação Rápida

Todos os arquivos foram criados? ✅
```bash
ls -lh benchmarking/quality/run_*.sh
ls -lh benchmarking/quality/*.md
```

Scripts são executáveis? ✅
```bash
test -x benchmarking/quality/run_quality_comparison_to_50k.sh && echo "✓"
test -x benchmarking/quality/run_article_quality_to_50k.sh && echo "✓"
```

---

## 📋 Próximos Passos

### Você está em qual situação?

**A) Quer rodar AGORA nesta máquina:**
→ Siga [QUICKSTART.md](QUICKSTART.md), seção "TL;DR"

**B) Quer rodar depois em outra máquina:**
→ Compartilhe os 2 scripts + [SETUP_AND_RUN.md](SETUP_AND_RUN.md) + [QUICKSTART.md](QUICKSTART.md)

**C) Quer customização avançada:**
→ Use [run_article_quality_to_50k.sh](run_article_quality_to_50k.sh) + [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Parte 9

**D) Quer só entender antes de rodar:**
→ Leia [CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md) + [QUICKSTART.md](QUICKSTART.md) Seção "Detalhes dos Scripts"

---

## 📞 Ficou com dúvida?

| Dúvida | Procure em |
|--------|-----------|
| "Como instalo em máquina nova?" | [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Part 2 |
| "Como executo?" | [QUICKSTART.md](QUICKSTART.md) Seção 2 |
| "Qual script devo usar?" | [CREATED_FILES_SUMMARY.md](CREATED_FILES_SUMMARY.md) |
| "Como interpreto resultados?" | [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Part 5 |
| "Tá demorando muito" | [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Part 6 |
| "Deu erro!" | [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Part 7 |
| "Parou no meio" | [QUICKSTART.md](QUICKSTART.md) Seção "Retomar Execução" |

---

## ✨ Resumo

✅ **Status:** Tudo pronto  
✅ **Scripts:** 2 (ambos executáveis)  
✅ **Documentação:** 4 arquivos (.md)  
✅ **Pronto para:** Outra máquina ou execução imediata  

🎯 **Próximo:** Escolha seu caminho acima e comece!

---

**Última atualização:** 12 de Agosto de 2026
