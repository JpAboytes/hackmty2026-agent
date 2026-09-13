"""The tool identities this orchestrator reasons about.

Three sets with three different jobs, deliberately kept apart:

* ``FINANCIAL_DOMAIN_TOOL_NAMES`` - the capabilities the deterministic
  financial router may plan a read against.
* ``DISCOVERY_TOOL_NAMES`` - how a model *reaches* a capability. A discovery
  contract, not an allowlist: it says nothing about what may execute.
* ``SCOPED_TOOL_NAMES`` - the security boundary. Every tool here reads
  user-owned rows, so its identity fields are overwritten with the UUID the
  authentication boundary produced.
"""

from __future__ import annotations

from collections.abc import Iterable

FINANCIAL_DOMAIN_TOOL_NAMES = frozenset(
    {
        "get_financial_overview",
        "get_accounts",
        "get_transactions",
        "analyze_spending",
        "get_cash_flow",
        "get_budget_progress",
        "get_savings_progress",
        "get_debt_overview",
        "get_upcoming_payments",
        "get_financial_alerts",
        "get_bank_statements",
        "get_payment_activity",
        "get_beneficiaries",
        "get_transaction_disputes",
        "compare_debt_scenarios",
    }
)

# The MCP server replaced its `tools/list` with progressive discovery, so the
# model receives these two synthetic tools and finds everything else through
# them. This is a discovery contract, not an allowlist: it says nothing about
# what may execute, only how a model reaches a capability.
SEARCH_TOOL_NAME = "search_tools"
CALL_TOOL_NAME = "call_tool"
DISCOVERY_TOOL_NAMES = frozenset({SEARCH_TOOL_NAME, CALL_TOOL_NAME})


def addressable_financial_tool_names(advertised_names: Iterable[str]) -> frozenset[str]:
    """Financial capabilities reachable through the current MCP catalog.

    A capability is addressable when it is advertised directly, or when the
    advertised ``call_tool`` proxy can address the hidden domain catalog. An
    unrelated non-empty tool list proves neither condition.
    """
    advertised = frozenset(advertised_names)
    if CALL_TOOL_NAME in advertised:
        return FINANCIAL_DOMAIN_TOOL_NAMES
    return FINANCIAL_DOMAIN_TOOL_NAMES & advertised


# Security boundary, kept deliberately separate from discovery: every tool here
# reads user-owned rows, so its identity fields are overwritten with the UUID
# the authentication boundary produced. A tool reaching MCP without passing
# through this set would be trusting model-supplied identity.
SCOPED_TOOL_NAMES = frozenset(
    {
        "select_rows",
        "visualize_allowed_data",
        "a2ui_action",
        "a2ui_form",
        *FINANCIAL_DOMAIN_TOOL_NAMES,
    }
)
