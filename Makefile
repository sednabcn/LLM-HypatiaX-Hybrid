# One-command flows. `make dryrun` proves the plumbing on SYNTHETIC data; `make real` is the only path to paper numbers.
PY ?= python3
RESULTS ?= results
.PHONY: test dryrun real paper paper-dryrun refs audit clean help
help:
	@echo "make test | dryrun | real RESULTS=<dir> | refs | audit | clean"
test:
	$(PY) -m pytest -q rsc/experiments/tests
dryrun:                       ## synthetic end-to-end: synth -> ingest -> audit -> tables/figures -> PDF (watermarked)
	rm -rf /tmp/hx_dry && $(PY) rsc/tools/synth_results.py /tmp/hx_dry
	$(PY) scripts/ingest_results.py /tmp/hx_dry --dryrun --reset
	$(PY) scripts/ingest_sweeps.py /tmp/hx_dry/sweeps/sweeps.csv --dryrun --reset-sweeps
	$(PY) rsc/experiments/analytics/audit_checks.py --dryrun
	$(PY) rsc/experiments/analytics/build.py --dryrun
	$(MAKE) paper-dryrun
real:                         ## real results only: refuses synthetic files
	$(PY) scripts/ingest_results.py $(RESULTS) --reset
	[ ! -f $(RESULTS)/sweeps/sweeps.csv ] || $(PY) scripts/ingest_sweeps.py $(RESULTS)/sweeps/sweeps.csv --reset-sweeps
	$(PY) rsc/experiments/analytics/audit_checks.py
	$(PY) rsc/experiments/analytics/build.py
	$(MAKE) paper
paper paper-dryrun:
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex | grep -E "^!|undefined" || true
	@echo "-> paper/main.pdf ($$(grep -c . paper/gen_paths.tex) path macros; status: $$(grep datastatus paper/gen_paths.tex))"
refs:                         ## needs internet (Crossref/arXiv)
	$(PY) scripts/validate_refs.py paper/references.bib --fix --strict
audit:
	$(PY) rsc/experiments/analytics/audit_checks.py
clean:
	cd paper && latexmk -C >/dev/null 2>&1; rm -rf _dryrun; rm -f rsc/db/results_dryrun.sqlite
