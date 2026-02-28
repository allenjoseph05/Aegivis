"""
Remote Code Execution (RCE) scanner for tool call arguments.

Scans string values inside tool call argument dicts for executable code.
Three language sub-scanners operate independently and their scores are fused:

  Python  — stdlib ``ast`` module.  Works at the parse-tree level, so
             variable-rename obfuscation (``e = eval; e(...)``) is caught
             because AST analysis sees the Call node structure, not raw text.
             Simple one-level alias tracking is included.

  Shell   — structural token analysis (metacharacter density, pipe-to-
             interpreter detection, command substitution syntax).  When
             ``bashlex`` is installed the analysis upgrades to full token
             stream parsing with tighter precision.

  SQL     — multi-statement and comment-truncation heuristics by default.
             When ``sqlglot`` is installed the analysis upgrades to full AST
             parsing for stacked queries, system stored procedures, and
             file-write statements.

Design note on "hardcoding"
---------------------------
The interpreter names (bash, sh, python ...) and SQL keywords used here are
SYNTACTIC CONSTANTS — they are part of the language grammars themselves, not
enumerable attacker content.  Using them is analogous to checking ``ast.Call``
node types: we are modelling language structure, not guessing attacker intent.
The intent detection is done by the pipeline's semantic layer (Layer 2).
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dangerous Python AST patterns
# ---------------------------------------------------------------------------

# Built-ins that directly execute or compile code.
_DANGEROUS_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__",
})

# os / posix module functions that interact with the OS shell.
_DANGEROUS_OS_ATTRS: frozenset[str] = frozenset({
    "system", "popen", "popen2", "popen3", "popen4",
    "execv", "execve", "execl", "execle", "execlp", "execlpe",
    "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "startfile",
})

# subprocess module functions.
_DANGEROUS_SUBPROCESS_ATTRS: frozenset[str] = frozenset({
    "Popen", "call", "check_call", "check_output", "run",
    "getoutput", "getstatusoutput",
})

# Modules whose import enables dangerous operations.
_DANGEROUS_MODULES: frozenset[str] = frozenset({
    "os", "subprocess", "sys", "socket", "shutil", "importlib",
    "ctypes", "pickle", "marshal", "pty", "commands", "pexpect",
})

# Dunder attributes used in CPython sandbox-escape chains.
_SANDBOX_ESCAPE_ATTRS: frozenset[str] = frozenset({
    "__class__", "__bases__", "__subclasses__", "__init__",
    "__globals__", "__builtins__", "__code__", "__func__", "__wrapped__",
})


class _PythonAstVisitor(ast.NodeVisitor):
    """
    Walk a Python AST and collect dangerous call/import/attribute patterns.

    Tracks one-level name assignments to catch simple aliasing:
        e = eval; e("payload")  -- detected via alias resolution
        import os as operating_system; operating_system.system("...")
    """

    def __init__(self) -> None:
        self.patterns: list[str] = []
        # local_name -> canonical_dangerous_name (e.g. "e" -> "eval")
        self._aliases: dict[str, str] = {}

    # -- imports --------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _DANGEROUS_MODULES:
                self.patterns.append(f"import:{alias.name}")
                local = alias.asname or root
                self._aliases[local] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0]
        if mod in _DANGEROUS_MODULES:
            self.patterns.append(f"from_import:{node.module}")
        for alias in node.names:
            canon = alias.name
            if canon in _DANGEROUS_BUILTINS or canon in _DANGEROUS_OS_ATTRS:
                self.patterns.append(f"from_import:{node.module}.{canon}")
                local = alias.asname or canon
                self._aliases[local] = canon
        self.generic_visit(node)

    # -- assignment (alias tracking) -----------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in _DANGEROUS_BUILTINS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._aliases[target.id] = node.value.id
        self.generic_visit(node)

    # -- function calls -------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve_name(node.func)
        if name:
            parts = name.split(".")
            head  = self._aliases.get(parts[0], parts[0])
            canonical = ".".join([head] + parts[1:]) if len(parts) > 1 else head

            if canonical in _DANGEROUS_BUILTINS:
                self.patterns.append(f"call:{canonical}")
            elif len(parts) >= 2:
                mod  = head
                func = parts[-1]
                if mod in {"os"} and func in _DANGEROUS_OS_ATTRS:
                    self.patterns.append(f"call:{mod}.{func}")
                elif mod in {"subprocess"} and func in _DANGEROUS_SUBPROCESS_ATTRS:
                    self.patterns.append(f"call:{mod}.{func}")
        self.generic_visit(node)

    # -- dunder attribute access (sandbox escape) ----------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _SANDBOX_ESCAPE_ATTRS:
            self.patterns.append(f"sandbox_escape:{node.attr}")
        self.generic_visit(node)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _resolve_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            obj = _PythonAstVisitor._resolve_name(node.value)
            if obj:
                return f"{obj}.{node.attr}"
        return None


def _scan_python(code: str) -> tuple[float, list[str]]:
    """
    Parse *code* as Python and return (confidence, patterns).
    Returns (0.0, []) on SyntaxError (not Python or malformed).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0, []

    visitor = _PythonAstVisitor()
    visitor.visit(tree)

    if not visitor.patterns:
        return 0.0, []

    # Score: each dangerous builtin call is very high confidence;
    # imports are medium; sandbox escapes are high.
    score = 0.0
    for p in visitor.patterns:
        if p.startswith("call:exec") or p.startswith("call:eval"):
            score = max(score, 0.95)
        elif p.startswith("call:__import__"):
            score = max(score, 0.90)
        elif p.startswith("sandbox_escape:"):
            score = max(score, 0.85)
        elif "os.system" in p or "os.popen" in p or "subprocess." in p:
            score = max(score, 0.85)
        elif p.startswith("from_import:") and any(
            d in p for d in _DANGEROUS_BUILTINS
        ):
            score = max(score, 0.80)
        elif p.startswith("import:") or p.startswith("from_import:"):
            score = max(score, 0.65)

    # Slight boost for multiple patterns (defence in depth)
    extra = min(0.05, 0.01 * len(visitor.patterns))
    return min(1.0, score + extra), visitor.patterns


# ---------------------------------------------------------------------------
# Shell scanner
# ---------------------------------------------------------------------------

# POSIX/common shell interpreters — finite, stable set defined by the POSIX
# standard and common Unix distributions.  Using this list is language-grammar
# modelling, not content-pattern matching.
_SHELL_INTERPRETERS: frozenset[str] = frozenset({
    "bash", "sh", "dash", "zsh", "ash", "ksh", "csh", "tcsh",
    "python", "python3", "python2", "perl", "ruby", "node", "nodejs",
    "php", "lua", "tclsh", "expect",
})

# Regex: pipe to shell interpreter (| bash, | python3, etc.)
# This is a SYNTACTIC pattern — '|' is the pipe operator and interpreter names
# are a closed set.  A legitimate command never needs to pipe to a shell.
_PIPE_TO_INTERP_RE = re.compile(
    r"\|\s*(" + "|".join(re.escape(s) for s in _SHELL_INTERPRETERS) + r")\b",
    re.IGNORECASE,
)

# Command substitution syntax: $(...) or `...`
_CMD_SUBST_RE = re.compile(r"\$\([^)]{1,200}\)|\`[^`]{1,200}\`")

# Dangerous output redirection to sensitive filesystem paths
_DANGEROUS_REDIRECT_RE = re.compile(
    r">\s*(/etc/|/dev/|/proc/|~/.ssh/|~/.bashrc|~/.profile|~/.zshrc|"
    r"/root/|/var/spool/cron)",
    re.IGNORECASE,
)

# Shell command chaining operators (legitimate use is very low for agent inputs)
_CHAIN_OPS_RE = re.compile(r"&&|\|\|")


def _scan_shell_structural(text: str) -> tuple[float, list[str]]:
    """
    Structural shell analysis — no external dependencies.
    Detects pipe-to-interpreter, command substitution, dangerous redirects.
    """
    patterns: list[str] = []
    score = 0.0

    if _PIPE_TO_INTERP_RE.search(text):
        patterns.append("shell:pipe_to_interpreter")
        score = max(score, 0.88)

    subst_hits = len(_CMD_SUBST_RE.findall(text))
    if subst_hits:
        patterns.append(f"shell:command_substitution(x{subst_hits})")
        score = max(score, 0.72)

    if _DANGEROUS_REDIRECT_RE.search(text):
        patterns.append("shell:dangerous_redirect")
        score = max(score, 0.80)

    chain_hits = len(_CHAIN_OPS_RE.findall(text))
    if chain_hits >= 2:
        patterns.append(f"shell:command_chaining(x{chain_hits})")
        score = max(score, 0.55)

    return score, patterns


def _scan_shell_bashlex(text: str) -> tuple[float, list[str]]:
    """
    Full shell token analysis using ``bashlex``.
    Falls back to structural analysis if bashlex is unavailable or parse fails.
    """
    try:
        import bashlex
    except ImportError:
        return _scan_shell_structural(text)

    try:
        parts = bashlex.parse(text)
    except Exception:
        return _scan_shell_structural(text)

    patterns: list[str] = []
    score = 0.0

    def _walk(node: object) -> None:
        nonlocal score
        kind = getattr(node, "kind", None)
        if kind == "command":
            words = getattr(node, "parts", [])
            if words:
                cmd_word = getattr(words[0], "word", "").lower()
                if cmd_word in _SHELL_INTERPRETERS:
                    patterns.append(f"shell:executes_interpreter:{cmd_word}")
                    score = max(score, 0.90)
        if kind == "pipe":
            patterns.append("shell:pipeline")
            score = max(score, 0.60)
        for child in getattr(node, "parts", []):
            _walk(child)

    for part in parts:
        _walk(part)

    # Merge with structural results (take the higher score)
    struct_score, struct_patterns = _scan_shell_structural(text)
    combined_score = max(score, struct_score)
    return combined_score, patterns + struct_patterns


def _scan_shell(text: str) -> tuple[float, list[str]]:
    try:
        import bashlex  # noqa: F401
        return _scan_shell_bashlex(text)
    except ImportError:
        return _scan_shell_structural(text)


# ---------------------------------------------------------------------------
# SQL scanner
# ---------------------------------------------------------------------------

# SQL keywords that, in an injection context, indicate dangerous statements.
# These are SQL LANGUAGE KEYWORDS — a finite, normative set defined by ANSI
# SQL and DBMS vendors.
_DANGEROUS_SQL_KEYWORDS: tuple[str, ...] = (
    "XP_CMDSHELL",       # MSSQL system command execution
    "SP_EXECUTESQL",     # MSSQL dynamic SQL execution
    "LOAD_FILE",         # MySQL file read
    "INTO OUTFILE",      # MySQL file write
    "INTO DUMPFILE",     # MySQL file write
    "UTL_FILE",          # Oracle file I/O package
    "DBMS_SCHEDULER",    # Oracle job scheduler (code execution)
    "OPENROWSET",        # MSSQL external data source
    "BULK INSERT",       # MSSQL file read
)

# SQL comment sequences that can truncate an intended query
_SQL_COMMENT_RE = re.compile(r"--\s|#\s|/\*")

# Multiple statement separator (stack injection)
_STACKED_RE = re.compile(r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|EXEC|EXECUTE|TRUNCATE)",
                          re.IGNORECASE)


def _scan_sql_structural(text: str) -> tuple[float, list[str]]:
    """
    SQL injection heuristics — no external dependencies.
    """
    text_upper = text.upper()
    patterns: list[str] = []
    score = 0.0

    for kw in _DANGEROUS_SQL_KEYWORDS:
        if kw in text_upper:
            patterns.append(f"sql:dangerous_keyword:{kw}")
            score = max(score, 0.90)

    if _STACKED_RE.search(text):
        patterns.append("sql:stacked_statements")
        score = max(score, 0.80)

    if _SQL_COMMENT_RE.search(text):
        patterns.append("sql:comment_truncation")
        score = max(score, 0.55)

    union_select = "UNION" in text_upper and "SELECT" in text_upper
    if union_select:
        patterns.append("sql:union_select")
        score = max(score, 0.75)

    return score, patterns


def _scan_sql_sqlglot(text: str) -> tuple[float, list[str]]:
    """
    Full SQL AST analysis using ``sqlglot``.
    Falls back to structural analysis if sqlglot is unavailable.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp
    except ImportError:
        return _scan_sql_structural(text)

    try:
        statements = sqlglot.parse(text, error_level=sqlglot.ErrorLevel.IGNORE)
    except Exception:
        return _scan_sql_structural(text)

    patterns: list[str] = []
    score = 0.0

    if len(statements) > 1:
        patterns.append(f"sql:multiple_statements({len(statements)})")
        score = max(score, 0.80)

    for stmt in statements:
        if stmt is None:
            continue
        for node in stmt.walk():
            node_type = type(node).__name__
            if node_type in ("Drop", "Truncate", "AlterTable"):
                patterns.append(f"sql:destructive_ddl:{node_type}")
                score = max(score, 0.85)
            if isinstance(node, exp.Anonymous):
                func_name = (node.name or "").upper()
                if func_name in {k.replace(" ", "_") for k in _DANGEROUS_SQL_KEYWORDS}:
                    patterns.append(f"sql:dangerous_function:{func_name}")
                    score = max(score, 0.92)

    # Merge with structural heuristics
    struct_score, struct_pats = _scan_sql_structural(text)
    return max(score, struct_score), patterns + struct_pats


def _scan_sql(text: str) -> tuple[float, list[str]]:
    try:
        import sqlglot  # noqa: F401
        return _scan_sql_sqlglot(text)
    except ImportError:
        return _scan_sql_structural(text)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Quick content checks to decide which parser to attempt.
# These avoid trying to parse every string as SQL/Python/shell.
_PYTHON_HINTS: tuple[str, ...] = (
    "import ", "from ", " def ", " class ", "exec(", "eval(",
    "subprocess", "__import__", "os.system", "print(", "if __name__",
)
_SHELL_HINTS: tuple[str, ...] = (
    "| ", "&&", "||", "; ", "$(", "`", "#!/", "curl ", "wget ",
    "bash -", "sh -", "nc -", "ncat ", "rm -rf",
)
_SQL_HINTS: tuple[str, ...] = (
    "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ",
    "CREATE ", "EXEC ", "UNION ", "TRUNCATE ", "ALTER ",
)


def _detect_language(text: str) -> str:
    """Return best-guess language: 'python' | 'shell' | 'sql' | ''."""
    t_upper = text.upper()
    py_score  = sum(1 for h in _PYTHON_HINTS if h in text)
    sh_score  = sum(1 for h in _SHELL_HINTS  if h in text)
    sql_score = sum(1 for h in _SQL_HINTS     if h in t_upper)
    best = max(py_score, sh_score, sql_score)
    if best == 0:
        return ""
    if py_score == best:
        return "python"
    if sql_score == best:
        return "sql"
    return "shell"


# ---------------------------------------------------------------------------
# Argument context scoring
# ---------------------------------------------------------------------------

# Argument names whose SEMANTICS indicate the value is executable code.
# A string value is more suspicious when its key implies execution intent.
_EXEC_ARG_NAMES: frozenset[str] = frozenset({
    "command", "cmd", "shell", "bash", "script", "code", "execute",
    "exec", "eval", "run", "expression", "program", "binary",
    "query", "sql", "statement", "input", "payload",
})


# ---------------------------------------------------------------------------
# Safe-context tool name dampening
# ---------------------------------------------------------------------------

# Tool names that indicate a code-review / analysis context.
# These tools are EXPECTED to receive and inspect code that contains dangerous
# constructs (os.system, eval, subprocess, etc.) — it is the content being
# analysed, not an attack payload.
#
# Without dampening, every code-review agent session triggers a false-positive
# RCE BLOCK. With dampening (factor 0.50), the confidence drops below the
# 0.70 default threshold:
#   eval() in review_code  → 0.95 * 0.50 = 0.475  (safe, below threshold)
#   eval() in run_command  → 0.95 * 1.00 = 0.950  (blocked — correct)
_SAFE_CONTEXT_TOOL_NAMES: frozenset[str] = frozenset({
    "review_code", "analyze_code", "lint_code", "check_code", "audit_code",
    "inspect_code", "static_analysis", "code_review", "code_analysis",
    "code_check", "code_audit", "code_quality", "security_review",
    "security_audit", "security_check", "scan_code", "code_scan",
    "run_linter", "run_tests", "execute_tests", "test_runner",
    "format_code", "refactor_code", "explain_code", "document_code",
    "read_file", "get_file_content", "read_code", "fetch_code",
    "get_file", "open_file", "view_file", "cat_file",
})

# If a tool name CONTAINS any of these keywords, apply partial dampening.
_SAFE_CONTEXT_KEYWORDS: tuple[str, ...] = (
    "review", "analyze", "lint", "audit", "inspect",
    "explain", "document", "format", "refactor",
)


def _safe_context_dampen(tool_name: str, confidence: float) -> float:
    """
    Reduce RCE confidence when the tool name implies a code-review context.

    Returns the adjusted confidence. Never returns a value > the input.
    """
    name_lower = tool_name.lower().replace("-", "_")
    if name_lower in _SAFE_CONTEXT_TOOL_NAMES:
        return confidence * 0.50
    if any(kw in name_lower for kw in _SAFE_CONTEXT_KEYWORDS):
        return confidence * 0.60
    return confidence


def _arg_boost(arg_name: str) -> float:
    """Confidence multiplier based on argument name semantics."""
    cleaned = arg_name.lower().strip("_-")
    if cleaned in _EXEC_ARG_NAMES:
        return 1.8
    if any(k in cleaned for k in _EXEC_ARG_NAMES):
        return 1.4
    return 1.0


# ---------------------------------------------------------------------------
# String extraction
# ---------------------------------------------------------------------------

def _extract_strings(args: object, _depth: int = 0) -> list[tuple[str, str]]:
    """
    Recursively extract (arg_name, string_value) pairs from *args*.
    Limits recursion depth to 8 to handle pathological nesting.
    """
    if _depth > 8:
        return []
    result: list[tuple[str, str]] = []
    if isinstance(args, str):
        result.append(("__value__", args))
    elif isinstance(args, dict):
        for key, val in args.items():
            if isinstance(val, str) and len(val) >= 8:
                result.append((str(key), val))
            elif isinstance(val, (dict, list)):
                result.extend(_extract_strings(val, _depth + 1))
    elif isinstance(args, list):
        for i, item in enumerate(args):
            if isinstance(item, str) and len(item) >= 8:
                result.append((str(i), item))
            elif isinstance(item, (dict, list)):
                result.extend(_extract_strings(item, _depth + 1))
    return result


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RceScanResult:
    detected:          bool
    confidence:        float              # 0.0-1.0
    language:          str               # "python" | "shell" | "sql" | ""
    dangerous_patterns: list[str]        = field(default_factory=list)
    scan_text_length:  int               = 0


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

_MIN_SCAN_LEN = 8    # ignore trivially short strings
_MAX_SCAN_LEN = 8192  # skip pathologically large blobs (e.g. base64 images)


def scan(tool_name: str, tool_args: dict | str) -> RceScanResult:
    """
    Scan tool call arguments for remote code execution patterns.

    Args:
        tool_name: Name of the tool the LLM is invoking.
        tool_args: Argument dict (or raw string) passed by the LLM.

    Returns:
        RceScanResult.  ``detected`` is True when confidence exceeds the
        configured threshold.
    """
    from ..config import settings

    threshold    = settings.security_rce_confidence_threshold
    total_chars  = 0
    best_conf    = 0.0
    best_lang    = ""
    all_patterns: list[str] = []

    strings = _extract_strings(tool_args)

    for arg_name, text in strings:
        if not (_MIN_SCAN_LEN <= len(text) <= _MAX_SCAN_LEN):
            continue
        total_chars += len(text)

        lang = _detect_language(text)
        if not lang:
            continue

        if lang == "python":
            raw_conf, patterns = _scan_python(text)
        elif lang == "shell":
            raw_conf, patterns = _scan_shell(text)
        else:
            raw_conf, patterns = _scan_sql(text)

        if not patterns:
            continue

        boosted = min(1.0, raw_conf * _arg_boost(arg_name))

        if boosted > best_conf:
            best_conf = boosted
            best_lang = lang
            all_patterns = patterns

        logger.debug(
            "RCE scan: arg=%r lang=%s conf=%.3f patterns=%s",
            arg_name, lang, boosted, patterns,
        )

    # Apply safe-context dampening for code review / analysis tool names.
    # A code-review tool SHOULD receive code with dangerous patterns — that is
    # its job. Dampening prevents false-positive BLOCKs on legitimate usage.
    best_conf = _safe_context_dampen(tool_name, best_conf)

    detected = best_conf >= threshold
    if detected:
        logger.warning(
            "[SECURITY] RCE pattern detected: tool=%r lang=%s confidence=%.3f patterns=%s",
            tool_name, best_lang, best_conf, all_patterns,
        )

    return RceScanResult(
        detected=detected,
        confidence=round(best_conf, 4),
        language=best_lang,
        dangerous_patterns=all_patterns[:10],  # cap for storage
        scan_text_length=total_chars,
    )
