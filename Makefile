# Space-separated list of extra pip packages (can be overridden at build time)
EXTRA_PIP_PACKAGES ?=

.PHONY: install-algo-packages wrap-algorithm

install-algo-packages:
	@if [ -n "$(EXTRA_PIP_PACKAGES)" ]; then \
		echo "Installing extra algorithm packages:"; \
		for pkg in $(EXTRA_PIP_PACKAGES); do \
			echo "  - $$pkg"; \
			pip install "$$pkg"; \
		done; \
	else \
		echo "No EXTRA_PIP_PACKAGES specified; skipping extra installs."; \
	fi

wrap-algorithm:
	python -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
