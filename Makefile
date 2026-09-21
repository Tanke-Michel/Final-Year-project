# Paper build.
#
#   make          compile the draft, with TODO/NUM markers visible
#   make sync     pull the latest generated tables and figures from results/
#   make check    fail if any TODO or NUM marker remains
#   make clean

PAPER = paper
RESULTS = ../results/figures

all: sync
	pdflatex -interaction=nonstopmode $(PAPER).tex >/dev/null || true
	-bibtex $(PAPER) >/dev/null 2>&1
	pdflatex -interaction=nonstopmode $(PAPER).tex >/dev/null || true
	pdflatex -interaction=nonstopmode $(PAPER).tex | tail -20
	@echo "built $(PAPER).pdf"

# Tables and figures are GENERATED. Never edit them here — regenerate with
# src/figures.py and re-run this, or the paper will drift from the results.
sync:
	@mkdir -p tables figures
	@if [ -f $(RESULTS)/table1_scores.tex ]; then \
	  cp $(RESULTS)/table1_scores.tex tables/; echo "synced table1_scores.tex"; \
	else \
	  echo "WARNING: no table1_scores.tex yet — run src/figures.py first"; \
	  printf '%%%% placeholder until src/figures.py has been run\n\\begin{table}[t]\\centering\n\\caption{SCORE by arm.}\\label{tab:scores}\n\\begin{tabular}{l}\\toprule Pending results \\\\ \\bottomrule\\end{tabular}\n\\end{table}\n' > tables/table1_scores.tex; \
	fi
	@cp $(RESULTS)/*.pdf figures/ 2>/dev/null && echo "synced figures" || echo "WARNING: no figures yet"

check:
	@n=$$(grep -v '^\s*%' $(PAPER).tex | grep -c '\\TODO{\|\\NUM{' || true); \
	if [ "$$n" -gt 0 ]; then \
	  echo "FAIL: $$n drafting marker(s) remain in $(PAPER).tex"; \
	  grep -n '\\TODO{\|\\NUM{' $(PAPER).tex | grep -v ':\s*%' | head -20; \
	  exit 1; \
	else echo "PASS: no drafting markers remain"; fi
	@n=$$(grep -c 'TODO-VERIFY' references.bib || true); \
	if [ "$$n" -gt 0 ]; then \
	  echo "FAIL: $$n unverified bibliography entr(ies)"; exit 1; \
	else echo "PASS: bibliography verified"; fi

clean:
	rm -f *.aux *.log *.bbl *.blg *.out $(PAPER).pdf

.PHONY: all sync check clean
