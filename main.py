#pip install transformers tensorflow pymupdf

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import fitz  # PyMuPDF

# Carregar modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained("pierreguillou/bert-base-cased-squad-v1.1-portuguese")
model = AutoModelForQuestionAnswering.from_pretrained("pierreguillou/bert-base-cased-squad-v1.1-portuguese")
model.eval()

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# Função para responder pergunta com janela deslizante
def answer_question_from_chunks(question, context, max_chunk_len=512, stride=128):
    inputs = tokenizer(
        question,
        context,
        max_length=max_chunk_len,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_tensors="pt",
        padding="max_length"
    )

    best_score = float("-inf")
    best_answer = ""

    for i in range(inputs["input_ids"].shape[0]):
        input_ids = {k: v[i:i+1] for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = model(**input_ids)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_index = int(start_logits.argmax())
        end_index = int(end_logits.argmax())

        if end_index < start_index:
            continue

        score = start_logits[start_index] + end_logits[end_index]

        if score > best_score:
            best_score = score.item()
            answer_ids = input_ids["input_ids"][0][start_index:end_index + 1]
            best_answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return best_answer.strip()

# Uso
pdf_path = "doencas_respiratorias_cronicas.pdf"
text = extract_text_from_pdf(pdf_path)
question = "Como a asma é classificada de acordo com a gravidade?"

answer = answer_question_from_chunks(question, text)
print("Resposta:", answer)
