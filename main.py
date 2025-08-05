import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load API key from .env file
load_dotenv()

# Set up the Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define expert roles
neuroimaging_expert = [
    SystemMessage(content="You are a neuroimaging expert. Explain how AI can be used in brain imaging studies of stuttering."),
    HumanMessage(content="Please describe how AI is applied to neuroimaging in developmental stuttering research.")
]

stuttering_expert = [
    
    SystemMessage(content="You are a researcher who specializes in the scientific study of stuttering. Share your insights on the use of AI in this field."),
    HumanMessage(content="What are the key ethical and technical concerns when applying AI to study stuttering?")
]

ethics_expert = [
    SystemMessage(content="You are an AI ethics policy expert. Your job is to identify ethical risks of using AI in neuroscience research."),
    HumanMessage(content="What ethical principles should guide AI use in developmental stuttering research?")
]

developmental_neuro_expert = [
    SystemMessage(content="You are a developmental neuroscientist. Explain how children's brain development impacts stuttering and what AI could help uncover."),
    HumanMessage(content="What role does AI play in understanding developmental aspects of stuttering?")
]

speech_pathologist = [
    SystemMessage(content="You are a clinical speech-language pathologist. Reflect on how AI could support or harm clinical practice for stuttering."),
    HumanMessage(content="What are your concerns and expectations for AI in speech therapy for stuttering?")
]

# Coordinator prompts all agents and prints their responses
neuro_output = llm.invoke(neuroimaging_expert).content
print("\n Neuroimaging Expert:\n", neuro_output)

stutter_output = llm.invoke(stuttering_expert).content
print("\n Stuttering Expert:\n", stutter_output)

ethics_output = llm.invoke(ethics_expert).content
print("\n Ethics Expert:\n", ethics_output)

devneuro_output = llm.invoke(developmental_neuro_expert).content
print("\n Developmental Neuroscientist:\n", devneuro_output)

clinician_output = llm.invoke(speech_pathologist).content
print("\n Speech Pathologist:\n", clinician_output)

# Coordinator agent summarizes everything
coordinator = [
    SystemMessage(content="You are the Coordinator. Your job is to read input from 5 domain experts and write a final document summarizing ethical AI guidelines for stuttering research."),
    HumanMessage(content=f"""Summarize and combine the key points from each expert:

Neuroimaging Expert: {neuro_output}

Stuttering Expert: {stutter_output}

Ethics Expert: {ethics_output}

Developmental Neuroscientist: {devneuro_output}

Speech Pathologist: {clinician_output}

Write a final 5-point ethical guideline summary for using AI in stuttering research.
""")
]

final_response = llm.invoke(coordinator)

print("\nFinal Guidelines:\n")
print(final_response.content)
