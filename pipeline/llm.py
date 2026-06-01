from config import GROQ_API_KEY, MODEL_NAME


def get_llm(backend: str = "groq"):
    """
    Returns a LangChain-compatible LLM.
    backend="groq"   → Groq cloud API (recommended, fast, free tier)
    backend="ollama" → local Llama via Ollama (no internet needed)
    """
    if backend == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0,  # deterministic — required for clinical use
        )
    elif backend == "ollama":
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="llama3")
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'groq' or 'ollama'.")
