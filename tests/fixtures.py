from __future__ import annotations

from sec_interpreter.schemas import RuleExtractorInput

# SEC cybersecurity disclosure rule excerpt — 3 paragraphs, each ~800+ chars to force 3 chunks
SAMPLE_RULE_TEXT = """\
The Securities and Exchange Commission is adopting new rules to enhance and standardize disclosures regarding cybersecurity risk management, strategy, governance, and incidents by public companies that are subject to the reporting requirements of the Securities Exchange Act of 1934. The rules require registrants to disclose material cybersecurity incidents they experience on Form 8-K and Form 6-K within four business days of determining that an incident is material, with a limited delay available where the United States Attorney General determines that immediate disclosure would pose a substantial risk to national security or public safety. The Commission has determined that investors benefit from timely, decision-useful information about material cybersecurity incidents, and that a uniform four-business-day disclosure framework strikes the appropriate balance between investor protection and operational practicality for registrants operating complex information systems across multiple jurisdictions and business lines.

The final rules also require registrants to describe, on an annual basis in their Form 10-K and Form 20-F filings, their processes, if any, for assessing, identifying, and managing material risks from cybersecurity threats, as well as whether any risks from cybersecurity threats have materially affected or are reasonably likely to materially affect the registrant's business strategy, results of operations, or financial condition. Registrants must also disclose the board of directors' oversight of risks from cybersecurity threats, including whether any board member or board committee is responsible for oversight of such risks, and must describe management's role and expertise in assessing and managing material risks from cybersecurity threats, including the relevant experience of the individuals or teams responsible for managing cybersecurity risk within the organization.

Registrants should maintain documentation sufficient to support the materiality determination process for cybersecurity incidents, including contemporaneous records of the facts and circumstances considered, the timeline of the assessment process, the individuals involved in making the determination, and the ultimate conclusion reached regarding materiality. These records should be retained in a manner consistent with the registrant's existing recordkeeping obligations under applicable Exchange Act rules, including Rules 17a-3 and 17a-4 for broker-dealers and Rule 204-2 for investment advisers, and should be made available for examination upon request by Commission staff or self-regulatory organization examiners. The absence of adequate documentation may itself be considered relevant to an assessment of whether a registrant's disclosure controls and procedures are effective.\
"""


def sample_input_dict(strict_citations: bool = False) -> dict:
    return {"rule_text": SAMPLE_RULE_TEXT, "strict_citations": strict_citations}


def sample_output_dict() -> dict:
    """Valid RuleExtractorOutput JSON matching SAMPLE_RULE_TEXT (3 chunks → src:0, src:1, src:2)."""
    return {
        "rule_metadata": {
            "rule_title": "SEC Cybersecurity Risk Management, Strategy, Governance, and Incident Disclosure Rules",
            "release_number": "33-11216",
            "publication_date": "2023-07-26",
            "effective_date": "2023-09-05",
            "citations": ["src:0"],
        },
        "rule_summary": {
            "summary": (
                "The SEC has adopted rules requiring public companies to disclose material "
                "cybersecurity incidents within four business days of a materiality determination. "
                "Annual disclosures on Form 10-K must describe cybersecurity risk management "
                "processes and board oversight responsibilities. "
                "Registrants must maintain documentation supporting materiality determinations "
                "and retain those records consistent with Exchange Act recordkeeping obligations."
            ),
            "citations": ["src:0", "src:1", "src:2"],
        },
        "key_obligations": [
            {
                "obligation_id": "OBL-001",
                "obligation_text": (
                    "Registrants should disclose material cybersecurity incidents on Form 8-K "
                    "within four business days of determining materiality."
                ),
                "cited_sections": ["Form 8-K", "Securities Exchange Act of 1934"],
                "source_citations": ["src:0"],
            },
            {
                "obligation_id": "OBL-002",
                "obligation_text": (
                    "Registrants should describe cybersecurity risk management processes, "
                    "board oversight, and management expertise on an annual basis in Form 10-K."
                ),
                "cited_sections": ["Form 10-K"],
                "source_citations": ["src:1"],
            },
            {
                "obligation_id": "OBL-003",
                "obligation_text": (
                    "Registrants should maintain contemporaneous records supporting the "
                    "materiality determination process for cybersecurity incidents."
                ),
                "cited_sections": ["Exchange Act recordkeeping rules"],
                "source_citations": ["src:2"],
            },
        ],
        "affected_entity_types": [
            {
                "entity_type": "Public companies subject to Exchange Act reporting requirements",
                "citation": "src:0",
            }
        ],
        "compliance_impact_areas": [
            {
                "area": "Disclosure",
                "linked_obligation_ids": ["OBL-001", "OBL-002"],
                "citations": ["src:0", "src:1"],
            },
            {
                "area": "Recordkeeping",
                "linked_obligation_ids": ["OBL-003"],
                "citations": ["src:2"],
            },
            {
                "area": "Governance",
                "linked_obligation_ids": ["OBL-002"],
                "citations": ["src:1"],
            },
            {
                "area": "Risk Management",
                "linked_obligation_ids": ["OBL-001", "OBL-002"],
                "citations": ["src:0", "src:1"],
            },
        ],
        "assumptions": [
            {
                "assumption_text": "The registrant is a domestic public company subject to Exchange Act Section 13 or 15(d)",
                "reason": "The rule applies specifically to Exchange Act reporting companies; private companies are excluded",
                "citation": "src:0",
            }
        ],
    }


def sample_input_model(strict_citations: bool = False) -> RuleExtractorInput:
    return RuleExtractorInput.model_validate(sample_input_dict(strict_citations=strict_citations))
