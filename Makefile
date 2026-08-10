PY := /Users/aswathiranjith/Library/Caches/pypoetry/virtualenvs/sahai-7jPp93h6-py3.13/bin/python
CORE := $(CURDIR)/libs/sahai-core

.PHONY: help test test-core test-services test-research up down logs fmt check

help:
	@echo "make test           run every test in the monorepo"
	@echo "make up             start the local stack (docker compose)"
	@echo "make down           stop the stack"
	@echo "make smoke          end-to-end request against a running stack"
	@echo "make check          compose invariants + full test suite"

test: test-core test-services test-research

test-core:
	@echo "── sahai-core ─────────────────────────────────────────"
	@cd libs/sahai-core && PYTHONPATH=. $(PY) -m pytest tests -q

test-services:
	@for s in executor tracer session; do \
		echo "── services/$$s ───────────────────────────────────────"; \
		(cd services/$$s && PYTHONPATH=".:$(CORE)" $(PY) -m pytest tests -q) || exit 1; \
	done

# The original research package and training job, still on the Kaggle path.
test-research:
	@echo "── research (sahai/) ──────────────────────────────────"
	@$(PY) -m pytest tests -q

# `docker compose up -d` returns once containers are CREATED, not once they are
# READY. Without this wait, `make up && make smoke` races the gateway and fails
# with "connection reset by peer".
up:
	docker compose up --build -d
	@echo "waiting for the stack to become healthy..."
	@for i in $$(seq 1 60); do \
		if curl -sf -o /dev/null http://localhost:8080/health 2>/dev/null; then \
			echo "stack healthy after $${i}s"; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "stack did not become healthy in 60s. Current state:"; \
	docker compose ps; \
	echo ""; \
	echo "check: curl -s localhost:8080/health | python3 -m json.tool"; \
	exit 1

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=100

smoke:
	@$(PY) scripts/smoke_test.py

check:
	@docker compose config >/dev/null && echo "compose: valid"
	@$(MAKE) test
