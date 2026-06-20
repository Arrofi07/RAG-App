# advisor_config.py
#
# Configuration for the "Study in Germany" AI Advisor persona.
#
# Separating this from main.py means you can swap the entire advisor domain
# (the system prompt, the profile fields, the dropdown options) without
# touching any retrieval or storage logic.

from typing import Any

# ------------------------------------------------------------------
# Profile field definitions
# ------------------------------------------------------------------
# Each entry: (field_key, label, type, options_or_None, default, help_text)
# Used by the Streamlit profile form to auto-generate inputs.

PROFILE_FIELDS: list[dict] = [
    {
        "key":     "nationality",
        "label":   "Nationality / Country of origin",
        "type":    "text",
        "default": "",
        "help":    "Affects visa requirements and recognition of your credentials.",
    },
    {
        "key":     "current_country",
        "label":   "Country you currently live in",
        "type":    "text",
        "default": "",
        "help":    "Determines which German embassy handles your visa.",
    },
    {
        "key":     "education_level",
        "label":   "Current education level",
        "type":    "select",
        "options": [
            "High school student",
            "High school graduate",
            "Bachelor's student",
            "Bachelor's graduate",
            "Master's student",
            "Master's graduate",
            "PhD student",
        ],
        "default": "Bachelor's graduate",
        "help":    "Used to match you with the right degree programs.",
    },
    {
        "key":     "field_of_study",
        "label":   "Field / subject you want to study",
        "type":    "text",
        "default": "",
        "help":    "e.g. Computer Science, Mechanical Engineering, Business Administration",
    },
    {
        "key":     "target_degree",
        "label":   "Target degree in Germany",
        "type":    "select",
        "options": ["Bachelor's", "Master's", "PhD / Doctorate", "Language course", "Undecided"],
        "default": "Master's",
        "help":    "",
    },
    {
        "key":     "german_level",
        "label":   "German language level",
        "type":    "select",
        "options": ["None (complete beginner)", "A1", "A2", "B1", "B2", "C1", "C2 (native-like)"],
        "default": "None (complete beginner)",
        "help":    "Many Master's programs accept English; some require TestDaF or DSH.",
    },
    {
        "key":     "english_level",
        "label":   "English language level / certification",
        "type":    "text",
        "default": "",
        "help":    "e.g. IELTS 7.0, TOEFL 100, C1, Native speaker",
    },
    {
        "key":     "target_universities",
        "label":   "Universities you're interested in (comma-separated)",
        "type":    "text",
        "default": "",
        "help":    "e.g. TU Munich, KIT, RWTH Aachen, Heidelberg University",
    },
    {
        "key":     "target_cities",
        "label":   "Cities you'd prefer to study in",
        "type":    "text",
        "default": "",
        "help":    "e.g. Berlin, Munich, Hamburg — or leave blank if flexible",
    },
    {
        "key":     "intake_semester",
        "label":   "Target intake semester",
        "type":    "select",
        "options": [
            "Winter 2025/26",
            "Summer 2026",
            "Winter 2026/27",
            "Summer 2027",
            "Not decided yet",
        ],
        "default": "Winter 2026/27",
        "help":    "Winter semester starts Oct; Summer starts April.",
    },
    {
        "key":     "budget_monthly_eur",
        "label":   "Monthly budget (EUR)",
        "type":    "number",
        "default": 1000,
        "help":    "Average student cost of living in Germany: €900–€1,200/month.",
    },
    {
        "key":     "visa_status",
        "label":   "Visa / residence permit status",
        "type":    "select",
        "options": [
            "Not started yet",
            "Gathering documents",
            "Application submitted",
            "Appointment booked",
            "Visa approved",
            "Already in Germany",
            "EU citizen (no visa needed)",
        ],
        "default": "Not started yet",
        "help":    "",
    },
    {
        "key":     "application_status",
        "label":   "University application status",
        "type":    "select",
        "options": [
            "Just researching",
            "Shortlisting universities",
            "Preparing documents",
            "Applied — waiting",
            "Received offer(s)",
            "Enrolled",
        ],
        "default": "Just researching",
        "help":    "",
    },
    {
        "key":     "extra_notes",
        "label":   "Anything else the advisor should know",
        "type":    "textarea",
        "default": "",
        "help":    "Scholarships you're eyeing, special circumstances, concerns, etc.",
    },
]


# ------------------------------------------------------------------
# System prompt builder
# ------------------------------------------------------------------

def build_system_prompt(profile: dict[str, Any] | None) -> str:
    """
    Build the advisor system prompt, optionally personalised with the user's profile.

    The profile block is injected right after the persona definition so the LLM
    treats it as authoritative background context for every answer it gives.
    """

    persona = """You are an expert AI advisor specialising in helping international \
students study in Germany. You have deep knowledge of:

- German university types (Universität, Fachhochschule / HAW, Technische Universität)
- Admission requirements, blocked-account rules, and application portals \
(uni-assist, Hochschulstart / DoSV, direct applications)
- Visa and residence permit process (student visa § 16b AufenthG, \
student applicant visa § 16c AufenthG)
- Scholarships: DAAD, Deutschlandstipendium, Erasmus+, foundation scholarships
- TestDaF, DSH, Goethe-Zertifikat for German language proof
- Recognition of foreign qualifications (anabin database, KMK)
- Student life: health insurance (public Krankenkasse), semester ticket, \
student unions (Studentenwerk), housing (Wohnheim, WG)
- APS certificate requirement for applicants from China, Vietnam, and Mongolia
- German academic calendar and key application deadlines

Tone: Warm, encouraging, precise. Use concrete numbers and deadlines where \
relevant. If you don't know something for certain, say so and suggest where \
the student can verify (official sources, DAAD website, university pages).

When answering, always consider the student's specific situation \
(their profile is below). Tailor advice to their background — do not give \
generic advice if the profile provides enough context to be specific."""

    if not profile or not any(profile.values()):
        profile_block = (
            "\n\nUser profile: Not yet filled in. "
            "Ask the student for key details (nationality, target degree, field, "
            "German level) if needed to give personalised advice."
        )
    else:
        lines = []

        def _val(key: str) -> str:
            v = profile.get(key, "")
            return str(v).strip() if v else "—"

        lines.append(f"- Name: {_val('name')}")
        lines.append(f"- Nationality: {_val('nationality')}")
        lines.append(f"- Currently in: {_val('current_country')}")
        lines.append(f"- Education level: {_val('education_level')}")
        lines.append(f"- Field of study: {_val('field_of_study')}")
        lines.append(f"- Target degree in Germany: {_val('target_degree')}")
        lines.append(f"- German level: {_val('german_level')}")
        lines.append(f"- English level: {_val('english_level')}")
        lines.append(f"- Interested in universities: {_val('target_universities')}")
        lines.append(f"- Preferred cities: {_val('target_cities')}")
        lines.append(f"- Target intake: {_val('intake_semester')}")
        lines.append(f"- Monthly budget: €{_val('budget_monthly_eur')}")
        lines.append(f"- Visa status: {_val('visa_status')}")
        lines.append(f"- Application status: {_val('application_status')}")
        if profile.get("extra_notes"):
            lines.append(f"- Additional context: {_val('extra_notes')}")

        profile_block = "\n\nStudent profile:\n" + "\n".join(lines)

    return persona + profile_block