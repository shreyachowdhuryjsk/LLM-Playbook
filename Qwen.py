# ============================================================
# 1. Import The Required Library
# =============================

from transformers import pipeline

# =============================
# 2. Load Model
# =============================

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

# =============================
# 3. Simple Q&A (No Context)
# =============================

ques = "What is an ace?"
prompt = f"Answer the question.\nQuestion: {ques}\nAnswer:"
print("🔹 Prompt:\n", prompt)

response = pipe(prompt, max_new_tokens=50)
print("\n🔹 Response:\n", response[0]["generated_text"])

# =============================
# 4. Context-Injected Q&A
# =============================

context = """An ace is a serve that lands in the correct service box and is untouched by the opponent, immediately winning the point for the server. It is one of the most 
effective and impressive ways to score in tennis, as it leaves the receiver with no opportunity to return the ball. Aces are often the result of powerful, accurate serves 
that catch the opponent off guard or out of position. They are especially common among professional players with strong serving skills. Players like Novak Djokovic, Serena Williams, 
and Roger Federer are known for using aces strategically to dominate their service games and maintain pressure on their opponents."""

prompt = f"""Answer the question using the given context.
Context: {context}
Question: {ques}
Answer:"""

print("\n🔹 Prompt with Context:\n", prompt)

response = pipe(prompt, max_new_tokens=100)
print("\n🔹 Response with Context:\n", response[0]["generated_text"])