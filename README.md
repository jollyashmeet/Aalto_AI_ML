# Aalto_AI_ML
Aalto Summer School ’25: Multi-agent LLM system drafting ethical AI guidelines for neuroimaging and stuttering research.

# Multi-Agent LLM System: Neuroscience of Stuttering

This project is part of a summer school on AI & Machine Learning at Aalto University.  
It uses a multi-agent LLM system (built with LangChain and Groq API) to draft ethical policy guidelines for AI in neuroscience, specifically for stuttering research.

# Install dependencies
python3 -m pip install langchain langchain-groq groq python-dotenv

# Set your API key
# Create a .env file:
GROQ_API_KEY=your_key_here

# Create a test_groq_connection.py
This script is a simple test to make sure everything is set up correctly and that your Groq API key is working.

python3 test_groq_connection.py

- Loads your API key from the `.env` file using `dotenv`
- Initializes the `ChatGroq` model (LLM) from LangChain
- Sends a test prompt:  
  *"What are ethical issues in using AI for neuroscience of stuttering?"*
- Prints the model's response in the terminal

# the main multi-agent script (main.py)
This script sets up a multi-agent system using LangChain and Groq’s LLaMA3 model. Five domain-specific agents (neuroimaging expert, stuttering researcher, AI ethics expert, developmental neuroscientist, and clinical speech-language pathologist) are prompted to give their views on AI use in stuttering research. A final “Coordinator” agent receives their responses and produces a 5-point summary document with ethical guidelines.

The Groq API key is loaded from the .env file, and the script uses ChatGroq from LangChain to manage the conversation flow. Each agent’s output is passed to the next via simple variable chaining, simulating a structured multi-agent discussion.

Create five AI agents using the LangChain + Groq setup:
- Neuroimaging Expert
- Stuttering Expert
- Ethical Policy Expert
- Developmental Neuroscientist (focused on stuttering)
- Clinical Speech Pathologist

Each agent was given a role-specific task or question, and they communicated sequentially through a shared memory. This means that each agent saw the previous response before giving their own.
I implemented a coordinator function to simulate this flow of ideas across agents. Each agent's response was stored and passed along to the next, creating a structured and informed dialogue.
This coordination enables the agents to collectively build toward a final document or insight — in our case, to explore the neuroscience of stuttering using AI responsibly.

# Interactive round - expert debate simulation (interactive_round.py)
This script is an extended version of the main multi-agent system. Instead of having the experts speak once and passing their responses to a coordinator, this version simulates a debate-like exchange between agents, where each expert responds to what the others have said. This makes the interaction feel more realistic and dynamic.

What the script does:
It includes three experts:

- A Neuroimaging Expert who talks about AI in brain imaging for stuttering
- A Stuttering Researcher who raises research-specific concerns
- An Ethics Policy Expert who evaluates ethical risks

Each agent takes turns responding to the others:
First, all three give their initial statements
Then, each expert reads what the others said and responds or critiques it
This creates a second round that feels like an academic conversation

After both rounds, a coordinator agent reads all their statements and writes a final ethical guideline summary.

Why this version?
This round-based setup mimics real scientific peer discussions. It captures a more realistic, multi-perspective policy drafting process, useful when simulating how expert collaboration works in sensitive research topics like the neuroscience of stuttering.

## Consensus-Based Interactive Round (consensus.py)

In addition to the basic “one-shot” and “debate” versions, we implemented a consensus workflow where experts must explicitly agree or disagree with a draft and propose revisions. The script does the following:

1. **Collect Expert Inputs**  
   - Neuroimaging Expert  
   - AI Ethics Expert  
   - Stuttering Researcher  

2. **Draft Initial Principles**  
   A Coordinator agent reads those three raw inputs and drafts **five high-level ethical principles**.

3. **Explicit Agree/Disagree Reviews**  
   Each expert is prompted to **agree or disagree** with each numbered principle (and must disagree with at least one), providing a brief reason and a suggested rewording when they disagree.

4. **Final Consensus Guidelines**  
   The Coordinator ingests all three sets of agree/disagree reviews and produces a **final set of five consensus-based guidelines**.

This workflow ensures true collaboration: experts not only contribute their perspectives but also critique and refine each other’s ideas before the Coordinator issues the final document.


