# Makefile for LaTeX Compilation
# Deep Learning Approaches for Bearing Fault Diagnosis

# Main document
MAIN = main
# Bibliography file
BIB = references

# Compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean cleanall help view

# Default target
all: $(MAIN).pdf

# Full compilation (LaTeX -> BibTeX -> LaTeX -> LaTeX)
$(MAIN).pdf: $(MAIN).tex $(BIB).bib sections/*.tex
	@echo "=== First LaTeX compilation ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "\n=== BibTeX compilation ==="
	$(BIBTEX) $(MAIN)
	@echo "\n=== Second LaTeX compilation ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "\n=== Third LaTeX compilation (final) ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "\n=== Compilation complete! ==="
	@echo "Output: $(MAIN).pdf"

# Quick compilation (single pass, for testing)
quick: $(MAIN).tex
	@echo "=== Quick compilation (single pass) ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	@echo "Output: $(MAIN).pdf"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.synctex.gz
	rm -f sections/*.aux
	@echo "Clean complete."

# Clean everything including PDF
cleanall: clean
	@echo "Removing PDF output..."
	rm -f $(MAIN).pdf
	@echo "Clean all complete."

# View PDF (requires PDF viewer)
view: $(MAIN).pdf
	@if command -v evince > /dev/null; then \
		evince $(MAIN).pdf & \
	elif command -v okular > /dev/null; then \
		okular $(MAIN).pdf & \
	elif command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf & \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Help message
help:
	@echo "Makefile for LaTeX Compilation"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Full compilation (recommended)"
	@echo "  make quick    - Quick compilation (single pass, for testing)"
	@echo "  make clean    - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Compile and open PDF viewer"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex"
	@echo "  - bibtex"
	@echo "  - LaTeX packages listed in main.tex"
