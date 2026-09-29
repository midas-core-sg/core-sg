# Publicacao da versao 0.1.1 no PyPI

Este arquivo e um roteiro temporario para publicar o Core-SG. Execute os
comandos a partir da raiz do repositorio e nao avance quando algum check
falhar.

O fluxo recomendado tem duas publicacoes:

1. `0.1.1rc5` no TestPyPI, para validar as novas wheels;
2. `0.1.1` no PyPI oficial, somente depois da validacao do release candidate.

Versoes publicadas no PyPI e no TestPyPI nao podem ser substituidas. Nunca
reutilize uma versao ou uma tag depois de alterar o codigo.

## 1. Conferir branch e alteracoes locais

```bash
# Deve imprimir: develop
git branch --show-current

# Revise todos os arquivos modificados e nao rastreados.
git status --short

# Confira as alteracoes que serao publicadas.
git diff
```

Checklist:

- estar na branch `develop`;
- confirmar que as alteracoes de `pyproject.toml`, `test.yml` e `release.yml`
  estao presentes;
- revisar separadamente alteracoes de benchmarks, artigo e testes;
- nao incluir arquivos locais ou resultados gerados por engano;
- confirmar que nenhum segredo, token ou credencial foi adicionado.

O repositorio atualmente possui outras alteracoes locais. Nao use
`git add .` sem revisar tudo primeiro.

## 2. Preparar o release candidate 0.1.1rc5

Edite estes metadados:

- em `pyproject.toml`, defina `version = "0.1.1rc5"`;
- em `docs/source/conf.py`, mantenha `release` coerente com a versao publicada,
  caso a documentacao exiba esse valor.

Depois confira:

```bash
# Deve encontrar 0.1.1rc5 no pyproject.toml.
rg -n 'version =|release =' pyproject.toml docs/source/conf.py

# Detecta erros simples de whitespace no diff.
git diff --check

# Revise o diff final antes do commit.
git diff -- pyproject.toml .github/workflows/test.yml .github/workflows/release.yml
```

## 3. Executar validacoes locais

Use um ambiente virtual limpo, preferencialmente com Python 3.13 ou 3.14:

```bash
# Cria o ambiente virtual local.
python3 -m venv .venv-release

# Linux ou macOS: ativa o ambiente.
source .venv-release/bin/activate

# Windows PowerShell: use este comando no lugar do anterior.
# .\.venv-release\Scripts\Activate.ps1

# Atualiza as ferramentas e instala o projeto com dependencias de teste.
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Executa lint, formatacao e testes.
python -m ruff check .
python -m ruff format --check .
python -m pytest tests -v -ra

# Constroi sdist e wheel local e valida seus metadados.
python -m build
python -m twine check dist/*
```

No Windows PowerShell, coloque o extra entre aspas da mesma forma:

```powershell
python -m pip install -e ".[dev]"
```

Checklist:

- todos os comandos devem terminar com codigo de saida zero;
- `dist/` deve conter uma wheel e um arquivo `.tar.gz`;
- `twine check` deve informar `PASSED` para todos os arquivos;
- falhas no Python 3.13 ou 3.14 devem ser corrigidas antes da tag.

Os artefatos locais nao serao publicados pelo workflow. O GitHub Actions
reconstroi tudo a partir do commit associado a tag.

## 4. Commitar e enviar o release candidate

Adicione apenas os arquivos que realmente pertencem ao release. Exemplo para
os arquivos de empacotamento e CI:

```bash
git add pyproject.toml .github/workflows/test.yml .github/workflows/release.yml
```

Adicione individualmente outros arquivos revisados que devam participar da
versao. Em seguida:

```bash
# Confira exatamente o que entrara no commit.
git diff --cached --check
git diff --cached

# Cria e envia o commit.
git commit -m "chore(release): prepare v0.1.1rc5"
git push origin develop
```

Se `docs/source/conf.py` tiver sido atualizado, inclua-o explicitamente no
`git add`.

## 5. Conferir o workflow de testes

Abra no GitHub:

`Actions -> Tests -> execucao do commit enviado`

O job `test` deve passar nas 15 combinacoes:

- Ubuntu com Python 3.10, 3.11, 3.12, 3.13 e 3.14;
- Windows com Python 3.10, 3.11, 3.12, 3.13 e 3.14;
- macOS com Python 3.10, 3.11, 3.12, 3.13 e 3.14.

Os jobs de qualidade, build e cobertura tambem devem passar. Nao crie a tag
enquanto algum job estiver falhando ou cancelado.

Opcionalmente, com o GitHub CLI:

```bash
# Mostra as execucoes recentes.
gh run list --workflow test.yml --limit 5

# Acompanhe a execucao correspondente ao commit.
# Substitua RUN_ID pelo identificador mostrado no comando anterior.
gh run watch RUN_ID --exit-status
```

## 6. Conferir Trusted Publishing do TestPyPI

No TestPyPI, confira o publisher associado ao projeto `core-sg`:

- owner/organizacao: `midas-core-sg`;
- repositorio: `core-sg`;
- workflow: `release.yml`;
- environment: `testpypi`.

No GitHub, abra `Settings -> Environments` e confirme que o environment
`testpypi` existe. O workflow usa OIDC (`id-token: write`), portanto nao deve
ser necessario armazenar usuario, senha ou token do TestPyPI.

## 7. Criar e enviar a tag do release candidate

Antes da tag, confirme que o commit local e o remoto sao o mesmo:

```bash
git fetch origin
git status
git rev-parse HEAD
git rev-parse origin/develop
```

Os dois hashes devem ser iguais e o `git status` nao deve indicar commits
locais ainda nao enviados.

Crie a tag somente depois dessas verificacoes:

```bash
# Cria uma tag anotada no commit atual.
git tag -a v0.1.1rc5 -m "core-sg v0.1.1rc5"

# Confere a tag antes de envia-la.
git show --stat v0.1.1rc5

# O push da tag inicia o workflow de release.
git push origin v0.1.1rc5
```

## 8. Validar o release candidate no GitHub Actions

Abra `Actions -> Release -> execucao da tag v0.1.1rc5`.

Confira:

- `release-context` escolheu `target: testpypi`;
- `build-sdist` passou;
- as wheels Linux x86_64 foram construidas e testadas;
- as wheels Windows AMD64 foram construidas e testadas;
- as wheels macOS x86_64 e arm64 foram construidas e testadas;
- cada wheel passou pela suite `pytest` depois de ser instalada;
- `publish-testpypi` passou;
- as 19 execucoes de `smoke-test-testpypi` instalaram a wheel publicada com
  `--only-binary=core-sg` e executaram o pacote com Python 3.10 a 3.14 em
  Ubuntu, Windows e macOS Apple Silicon, e 3.10 a 3.13 em macOS Intel.

Sao esperadas 19 wheels, alem do sdist: Python 3.10 a 3.14 em Linux, Windows e
macOS Apple Silicon, e Python 3.10 a 3.13 em macOS Intel. Confira os artefatos
do workflow e os arquivos da versao no TestPyPI.

Nao avance para o PyPI oficial se qualquer wheel esperada estiver ausente.

## 9. Testar a instalacao binaria do TestPyPI

Execute estes testes em ambientes virtuais limpos no Ubuntu, Windows e macOS.
Quando possivel, teste tanto Python 3.13 quanto 3.14.

```bash
# Crie e ative um ambiente virtual limpo antes deste comando.
python -m pip install --upgrade pip

# Exige uma wheel para core-sg; nao permite fallback para compilacao local.
python -m pip install --no-cache-dir \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --only-binary=core-sg \
  core-sg==0.1.1rc5

# Confirma a versao e o local de importacao.
python -c "import importlib.metadata as m, core_sg; print(m.version('core-sg')); print(core_sg.__file__)"
```

No Windows PowerShell, escreva o comando em uma linha ou use crase como
continuacao de linha:

```powershell
python -m pip install --no-cache-dir --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ --only-binary=core-sg core-sg==0.1.1rc5
```

Smoke test funcional:

```bash
python -c "from sklearn.datasets import make_blobs; from core_sg import CoreSG; X,_=make_blobs(n_samples=120,n_features=3,centers=3,random_state=42); model=CoreSG(metric='euclidean',p=2); model.fit(X,k_max=5); assert model.extract_mst_from_core_sg(k=3) is not None; print('OK')"
```

Checklist:

- a versao impressa deve ser `0.1.1rc5`;
- o caminho de `core_sg.__file__` deve estar dentro do ambiente virtual;
- a instalacao nao deve executar Cython nem invocar compilador C/C++;
- o smoke test deve imprimir `OK`.

## 10. Preparar a versao final 0.1.1

Somente depois de todos os checks do release candidate:

- altere `version = "0.1.1"` em `pyproject.toml`;
- atualize `docs/source/conf.py`, se esse campo for mantido manualmente;
- atualize changelog ou notas de release, se aplicavel;
- nao altere o codigo funcional entre `rc5` e a versao final. Se houver uma
  correcao funcional, publique primeiro outro RC, como `0.1.1rc6`.

Confira e envie:

```bash
rg -n 'version =|release =' pyproject.toml docs/source/conf.py
git diff --check
git diff

# Ajuste a lista caso outros arquivos de release tenham sido atualizados.
git add pyproject.toml docs/source/conf.py
git diff --cached --check
git diff --cached
git commit -m "chore(release): prepare v0.1.1"
git push origin develop
```

Espere novamente todos os jobs do workflow `Tests`. Mesmo sendo apenas uma
mudanca de versao, a publicacao final deve partir de um commit verde.

## 11. Conferir Trusted Publishing do PyPI oficial

No PyPI oficial, confira o publisher associado ao projeto `core-sg`:

- owner/organizacao: `midas-core-sg`;
- repositorio: `core-sg`;
- workflow: `release.yml`;
- environment: `pypi`.

No GitHub, confirme que o environment `pypi` existe. O publisher do PyPI e
independente do publisher do TestPyPI; configurar um nao configura o outro.

Se o environment tiver aprovadores obrigatorios, deixe uma pessoa disponivel
para aprovar o job. Nao adicione token manual se o Trusted Publishing estiver
corretamente configurado.

## 12. Criar e enviar a tag final

```bash
# Sincroniza e confirma que a tag apontara para origin/develop.
git fetch origin
git status
git rev-parse HEAD
git rev-parse origin/develop

# Confirma a versao final antes da operacao irreversivel.
rg -n '^version = "0\.1\.1"$' pyproject.toml

# Cria e inspeciona a tag final.
git tag -a v0.1.1 -m "core-sg v0.1.1"
git show --stat v0.1.1

# Este comando inicia a publicacao no PyPI oficial.
git push origin v0.1.1
```

O workflow confere que a tag pertence a `develop` e que `v0.1.1` corresponde
a `version = "0.1.1"`. Como a tag nao contem `rc`, o destino sera o PyPI
oficial.

## 13. Conferir a publicacao final

No workflow `Release`, confirme:

- `release-context` escolheu `target: pypi`;
- todos os builds e testes de wheels passaram novamente;
- `publish-pypi` passou;
- as 19 execucoes de `smoke-test-pypi` instalaram somente wheels e passaram
  com Python 3.10 a 3.14 em Ubuntu, Windows e macOS Apple Silicon, e 3.10 a
  3.13 em macOS Intel.

Depois teste a instalacao publicada em um ambiente virtual novo:

```bash
python -m venv .venv-pypi-check
source .venv-pypi-check/bin/activate

# Windows PowerShell:
# .\.venv-pypi-check\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Garante que core-sg seja instalado como wheel, sem compilacao local.
python -m pip install --no-cache-dir --only-binary=core-sg core-sg==0.1.1

# Confere versao e origem do pacote.
python -c "import importlib.metadata as m, core_sg; print(m.version('core-sg')); print(core_sg.__file__)"

# Executa o smoke test final.
python -c "from sklearn.datasets import make_blobs; from core_sg import CoreSG; X,_=make_blobs(n_samples=120,n_features=3,centers=3,random_state=42); model=CoreSG(metric='euclidean',p=2); model.fit(X,k_max=5); assert model.extract_mst_from_core_sg(k=3) is not None; print('OK')"
```

Confira tambem na pagina do PyPI:

- versao exibida: `0.1.1`;
- README renderizado corretamente;
- classificadores incluem Python 3.13 e 3.14;
- arquivos incluem sdist e todas as wheels esperadas;
- links de Source, Issues e Documentation funcionam.

## 14. Finalizar o fluxo Git

Depois da publicacao final, siga a politica do projeto para levar o commit da
versao estavel de `develop` para `main`. Prefira pull request e confirme que os
checks continuam verdes.

Nao apague nem recrie a tag publicada. Se um problema for descoberto depois da
publicacao, prepare uma nova versao, por exemplo `0.1.2`.

Quando todo o processo terminar, este arquivo pode ser removido em um commit
separado ou transformado em documentacao permanente de release.
