import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ——————————————————————————————
# Setup
# ——————————————————————————————
load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7  # higher creativity for review step
)

# ——————————————————————————————
# 0) Define each expert’s system + user prompt
# ——————————————————————————————
neuroimaging_expert = [
    SystemMessage(content="You are a neuroimaging expert."),
    HumanMessage(content="Explain how AI is applied in neuroimaging studies of developmental stuttering.")
]
ethics_expert = [
    SystemMessage(content="You are an AI ethics policy expert."),
    HumanMessage(content="What ethical principles should guide AI use in stuttering research?")
]
stuttering_expert = [
    SystemMessage(content="You are a researcher specializing in stuttering."),
    HumanMessage(content="As a researcher, what technical or ethical issues matter most when using AI for stuttering?")
]

# ——————————————————————————————
# 1) First round: collect raw inputs
# ——————————————————————————————
neuro_output   = llm.invoke(neuroimaging_expert).content
ethics_output  = llm.invoke(ethics_expert).content
stutter_output = llm.invoke(stuttering_expert).content

print("\n--- Initial Expert Inputs ---")
print("\n[Neuroimaging]\n", neuro_output)
print("\n[Ethics]\n", ethics_output)
print("\n[Stuttering Research]\n", stutter_output)

# ——————————————————————————————
# 2) Coordinator drafts initial principles
# ——————————————————————————————
draft = llm.invoke([
    SystemMessage(content="You are the Coordinator."),
    HumanMessage(content=(
        "Based on these expert inputs, draft five high-level principles for ethical AI in stuttering research:\n\n"
        f"- {neuro_output}\n\n"
        f"- {ethics_output}\n\n"
        f"- {stutter_output}"
    ))
]).content

print("\n--- Initial Draft Principles ---\n", draft)

# ——————————————————————————————
# 3) Experts review the draft with explicit Agree/Disagree
# ——————————————————————————————
revisions = []
roles_and_experts = [
    ("Neuroimaging", neuroimaging_expert),
    ("Ethics",       ethics_expert),
    ("Stuttering",   stuttering_expert)
]

for role, expert in roles_and_experts:
    review = llm.invoke([
        SystemMessage(content=f"You are the {role} Expert."),
        HumanMessage(content=(
            f"Here is the draft principles:\n\n{draft}\n\n"
            "For each numbered principle, respond in this exact format:\n"
            "1. Agree or Disagree: [brief reason]\n"
            "2. Agree or Disagree: [brief reason]\n"
            "3. Agree or Disagree: [brief reason]\n"
            "4. Agree or Disagree: [brief reason]\n"
            "5. Agree or Disagree: [brief reason]\n\n"
            "**You must Disagree with at least one principle** and suggest how to reword it."
        ))
    ]).content
    revisions.append((role, review))
    print(f"\n--- {role} Review ---\n{review}")

# ——————————————————————————————
# 4) Coordinator finalizes consensus guidelines
# ——————————————————————————————
final = llm.invoke([
    SystemMessage(content="You are the Coordinator."),
    HumanMessage(content=(
        "Incorporate these expert reviews into a final set of five consensus-based "
        "ethical AI guidelines for stuttering research:\n\n" +
        "\n\n".join(f"{role} review:\n{rev}" for role, rev in revisions)
    ))
]).content

print("\n=== Final Consensus Guidelines ===\n", final)
