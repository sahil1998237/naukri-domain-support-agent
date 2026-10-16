"""
Task 15: Least-Autonomy Enforcement for Multi-Agent Governance (Day 12).

Verifies and audits the Principle of Least Privilege across all agents in the
CrewAI multi-agent hierarchy:
1. Retrieval Agent: Restricted exclusively to `retrieve_policy_context`. Zero access
   to candidate application records.
2. Lookup Agent: Restricted exclusively to `check_job_application_status`. Zero access
   to vector policy indices.
3. Response Composer Agent: Zero tools assigned. Operates solely on findings forwarded
   by upstream specialist agents.

Provides programmatic audit utilities and a formal governance justification.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.crew import build_naukri_support_crew


# ==============================================================================
# FORMAL LEAST-AUTONOMY GOVERNANCE EXPLANATION
# ==============================================================================

LEAST_AUTONOMY_EXPLANATION = (
    "In enterprise AI recruitment platforms, granting agents unrestricted or blanket tool "
    "access introduces critical data leakage risks, prompt-injection escalation vulnerabilities, "
    "and non-deterministic execution loops. Under our Least-Autonomy Multi-Agent Architecture, "
    "each agent operates strictly within the minimal authorization boundary required for its designated "
    "role: the Naukri Policy Retrieval Specialist holds read-only access to ChromaDB vector policy indices "
    "with zero access to candidate application records; the Application Status & Escalation Specialist "
    "holds single-record lookup and escalation privileges without policy retrieval access; and the "
    "Senior Candidate Support Composer has zero tool access, operating exclusively as a synthesizer of "
    "verified specialist findings. This strict segmentation guarantees that a prompt-injection attack "
    "or model hallucination in one agent cannot cascade into unauthorized database queries or "
    "cross-boundary candidate privacy violations."
)


@dataclass
class AgentToolAudit:
    agent_name: str
    role: str
    assigned_tools: List[str]
    tool_count: int
    authorized_data_domain: str
    is_compliant: bool
    compliance_notes: str


def audit_least_autonomy(crew: Optional[Any] = None) -> Dict[str, Any]:
    """
    Audits the active CrewAI multi-agent crew for compliance with the
    Principle of Least Privilege (Least Autonomy).
    """
    if crew is None:
        crew = build_naukri_support_crew()

    audits: List[AgentToolAudit] = []
    is_overall_compliant = True

    for agent in crew.agents:
        role = agent.role
        tool_names = [getattr(t, "name", str(t)) for t in (agent.tools or [])]
        tool_count = len(tool_names)

        if "Retrieval" in role or "Policy" in role:
            # Policy agent must ONLY have retrieve_policy_context
            expected_tool = "retrieve_policy_context"
            has_status_tool = any("status" in t.lower() or "application" in t.lower() for t in tool_names)
            compliant = (tool_count == 1 and expected_tool in tool_names and not has_status_tool)
            domain = "ChromaDB Policy Vector Knowledge Base (Read-Only)"
            notes = "Compliant: Read-only access to policies; isolated from candidate applications." if compliant else "Non-compliant tool assignment"

        elif "Lookup" in role or "Status" in role:
            # Status agent must ONLY have check_job_application_status
            expected_tool = "check_job_application_status"
            has_policy_tool = any("policy" in t.lower() or "retrieve" in t.lower() for t in tool_names)
            compliant = (tool_count == 1 and expected_tool in tool_names and not has_policy_tool)
            domain = "Candidate Application Records & Escalation Calculator (Single Record)"
            notes = "Compliant: Single-record application lookup; isolated from policy vectors." if compliant else "Non-compliant tool assignment"

        elif "Composer" in role:
            # Composer agent must have ZERO tools
            compliant = (tool_count == 0)
            domain = "Pure Synthesis & Communication (Zero Direct Data Access)"
            notes = "Compliant: Zero tools assigned. Operates solely on forwarded findings." if compliant else "Non-compliant: Composer must have zero tools"

        else:
            compliant = False
            domain = "Unknown"
            notes = "Unrecognized agent role"

        if not compliant:
            is_overall_compliant = False

        audits.append(
            AgentToolAudit(
                agent_name=agent.role,
                role=role,
                assigned_tools=tool_names,
                tool_count=tool_count,
                authorized_data_domain=domain,
                is_compliant=compliant,
                compliance_notes=notes,
            )
        )

    return {
        "is_overall_compliant": is_overall_compliant,
        "total_agents": len(audits),
        "audits": [asdict(a) for a in audits],
        "governance_explanation": LEAST_AUTONOMY_EXPLANATION,
    }


def format_autonomy_report(audit_result: Dict[str, Any]) -> str:
    """Formats the audit results into a readable governance report."""
    lines = [
        "=" * 80,
        "  NAUKRI.COM DOMAIN SUPPORT AGENT - LEAST-AUTONOMY GOVERNANCE AUDIT (TASK 15)",
        "=" * 80,
        f"Overall Compliance Status: {'PASSED (COMPLIANT)' if audit_result['is_overall_compliant'] else 'FAILED'}",
        f"Total Agents Inspected:   {audit_result['total_agents']}\n",
        "-" * 80,
        f"{'Agent Role':<40} | {'Tools':<5} | {'Assigned Toolset':<30}",
        "-" * 80,
    ]

    for item in audit_result["audits"]:
        tools_str = ", ".join(item["assigned_tools"]) if item["assigned_tools"] else "[None (Zero Tools)]"
        lines.append(f"{item['agent_name'][:40]:<40} | {item['tool_count']:<5} | {tools_str:<30}")
        lines.append(f"  Authorization Boundary: {item['authorized_data_domain']}")
        lines.append(f"  Status:                 {item['compliance_notes']}\n")

    lines.extend([
        "-" * 80,
        "LEAST-AUTONOMY ARCHITECTURAL JUSTIFICATION:",
        "-" * 80,
        audit_result["governance_explanation"],
        "=" * 80,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    res = audit_least_autonomy()
    report = format_autonomy_report(res)
    print(report)
