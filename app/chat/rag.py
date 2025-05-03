import os
from openai import OpenAI
from typing import List

from .documents import normalize


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_TOKENS = 15000


system_message = """"Você é um especialista em Counter-Strike com foco exclusivo na equipe da FURIA. Sua função é responder perguntas de torcedores sobre a história do time, seus jogadores, partidas, estatísticas e conquistas.

                    Regras de conduta:
                    - Responda apenas perguntas relacionadas ao time de CS da FURIA.
                    - Caso a pergunta esteja fora do escopo, informe educadamente que só responde sobre a FURIA.
                    - Se não houver informação suficiente, diga claramente que não possui dados suficientes para responder.
                    - Fale sempre como um especialista humano, sem mencionar que é um assistente virtual ou que está usando um contexto.
                    - Use uma linguagem clara, objetiva e acessível para torcedores brasileiros.

                    Seu objetivo é ajudar torcedores da FURIA com respostas completas e confiáveis.
                    """


def rag_query(user_query: str, docs: List[str], index, embedder, top_k=6):
    try:
        query_embedding = embedder.encode(
            [normalize(user_query)], convert_to_numpy=True
        )

        scores, I = index.search(query_embedding, top_k)
        threshold = 0.5
        filtered_results = [
            idx for idx, score in zip(I[0], scores[0]) if score >= threshold
        ]
        if len(filtered_results) > 0:
            retrieved_docs = [docs[i] for i in filtered_results]
        else:
            retrieved_docs = [docs[I[0][0]]]

        context = "\n\n".join(retrieved_docs)[:MAX_TOKENS]

        prompt = f"""
                    Informações disponíveis:
                    {context}

                    Pergunta:
                    {user_query}

                    Resposta:"""

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_message,
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
