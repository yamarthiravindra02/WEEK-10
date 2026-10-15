
!pip -q install langchain-google-genai

import os
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI

# Set API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0
)

problem = """
A farmer has 15 cows. He buys 7 more cows.
Then he sells half of his cows. How many cows does he have left?
"""

# Direct prompt
direct_prompt = problem + "\nGive only the final numerical answer."

# Chain-of-Thought prompt
cot_prompt = problem + """
Let's think step by step. Show the calculation briefly
and clearly state the final numerical answer.
"""

# Get responses
direct_answer = llm.invoke(direct_prompt).text
cot_answer = llm.invoke(cot_prompt).text

# Display results
print("=== DIRECT PROMPT ===")
print("Answer:", direct_answer)

print("\n=== CHAIN-OF-THOUGHT PROMPT ===")
print(cot_answer)

# Comparison
expected = 11
print("\n=== ANALYSIS ===")
print("Expected answer:", expected)
print("Direct answer correct:", str(expected) in direct_answer)
print("CoT answer correct:", str(expected) in cot_answer)
