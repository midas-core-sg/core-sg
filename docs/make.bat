@ECHO OFF

pushd %~dp0

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

if "%1" == "figures" (
	python scripts\generate_diagrams.py
	python scripts\generate_benchmark_figures.py
	goto end
)

if "%1" == "strict" (
	python scripts\generate_diagrams.py
	python scripts\generate_benchmark_figures.py
	%SPHINXBUILD% -W -b html %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS%
	goto end
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
