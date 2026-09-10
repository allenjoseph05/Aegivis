"""
Sandbox adapter implementations — Phase 16.x.

Shared utilities live in _common.py (tokenize function).

Implemented adapters:
  16.1  sql_adapter.py       — SQL dry-run (savepoint rollback + COUNT(*))
  16.2  e2b_adapter.py       — E2B Firecracker code sandbox
  16.3  aws_adapter.py       — AWS DryRun + IAM simulation
  16.4  file_adapter.py      — Filesystem glob-count
  16.5  http_adapter.py      — HTTP capture-and-hold
  16.6  email.py             — Email preview (recipient count, PII in body)
  16.7  git.py               — Git push --dry-run, status count, rm paths
  16.8  terraform.py         — Terraform plan: init + plan + show -json
  16.8  kubectl.py           — kubectl apply/delete --dry-run=client
"""
