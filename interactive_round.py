import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load API key from .env file
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Step 1: Neuroimaging expert goes first
neuroimaging_round = [
    SystemMessage(content="You are a neuroimaging expert. Share how AI is applied in brain imaging studies of developmental stuttering."),
    HumanMessage(content="Start the discussion by outlining how AI contributes to neuroimaging research in stuttering.")
]
neuro_output = llm.invoke(neuroimaging_round).content
print("[Neuroimaging Expert]:\n", neuro_output, "\n")

# Step 2: Ethics expert responds to neuroimaging expert
ethics_round = [
    SystemMessage(content="You are an AI ethics expert. Respond to the neuroimaging expert and raise ethical concerns about their use of AI."),
    HumanMessage(content=f"The neuroimaging expert said: {neuro_output}\nWhat are your ethical concerns or counterpoints?")
]
ethics_output = llm.invoke(ethics_round).content
print("[Ethics Expert]:\n", ethics_output, "\n")

# Step 3: Stuttering researcher responds to both
stuttering_round = [
    SystemMessage(content="You are a researcher specialising in stuttering. Respond to both the neuroimaging expert and the ethics expert. Offer a balanced view."),
    HumanMessage(content=f"The neuroimaging expert said: {neuro_output}\n\nThe ethics expert said: {ethics_output}\n\nWhat is your perspective on the use of AI in stuttering research?")
]
stutter_output = llm.invoke(stuttering_round).content
print("[Stuttering Researcher]:\n", stutter_output, "\n")

# Optional: Coordinator wraps up
coordinator_round = [
    SystemMessage(content="You are the Coordinator. Your job is to read all three expert inputs and write a 3-point ethical summary for AI in stuttering research."),
    HumanMessage(content=f"Summarize key points from this discussion:\n\nNeuroimaging Expert: {neuro_output}\n\nEthics Expert: {ethics_output}\n\nStuttering Researcher: {stutter_output}")
]
final_summary = llm.invoke(coordinator_round).content

print("[Coordinator Summary]:\n", final_summary)
