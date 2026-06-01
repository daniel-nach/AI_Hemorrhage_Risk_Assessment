def build_chain(llm, prompt):
    """
    Connects the prompt template to the LLM using LangChain's pipe operator.
    Calling chain.invoke({...}) fills the prompt then sends it to the LLM.
    """
    return prompt | llm


def process_patient(chain, patient_dict: dict, guidelines: str) -> str:
    """
    Runs the chain for a single patient and returns the recommendation text.
    """
    patient_str = "\n".join(f"{k}: {v}" for k, v in patient_dict.items())

    result = chain.invoke({
        "guidelines": guidelines,
        "patient_data": patient_str,
    })

    # ChatGroq returns a message object; .content extracts the text
    return result.content if hasattr(result, "content") else str(result)
