import os
from openai import OpenAI
from typing import List


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_TOKENS = 15000


def rag_query(user_query: str, docs: List[str], index, embedder, top_k=4):
    try:
        query_embedding = embedder.encode([user_query], convert_to_numpy=True)

        _, I = index.search(query_embedding, top_k)
        retrieved_docs = [docs[i] for i in I[0]]

        context = "\n\n".join(retrieved_docs)[:MAX_TOKENS]

        prompt = f"""Você é um assistente da torcida do time de CS da FURIA. Com base nas informações abaixo, responda de maneira clara e completa o que foi perguntado.
                    Caso as informações sejam insuficientes, responda que você não possui informação suficiente para responder o que foi perguntado.
                    Caso a pergunta não seja relacionada ao time de CS da Furia, responda que você apenas ajuda os torcedores com questões relacionadas ao time.

                    IMPORTANTE: Não faça menções às informações fornecidas, lembre-se: Você é o especialista.
                    Retorne APENAS sua resposta para a pergunta, não retorne partes da pergunta.

                    Informações:
                    {context}

                    Pergunta: {user_query}
                    Resposta:"""

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente especializado na equipe de CS da FURIA.",
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=500,
            temperature=0.7,
        )

        return completion.choices[0].message.content.strip()
    except Exception as err:
        print("Error in RAG: ", err)
        return "Ocorreu um erro, tente novamente..."
