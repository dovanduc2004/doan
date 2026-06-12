import yaml
from langchain_core.prompts import ChatPromptTemplate
from source.core.config import Settings
import textwrap
import os
import json
import re
from sentence_transformers import util
from source.model.embedding_model import Sentences_Transformer_Embedding
def load_prompt_from_yaml(settings: Settings, section: str) -> ChatPromptTemplate:
    # Tạo đường dẫn tuyệt đối đến file YAML
    current_dir = os.path.dirname(__file__)
    yaml_path = os.path.join(current_dir, '..', 'core', settings.YAML_PATH)
    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    messages = yaml_data['prompts'][section]['messages']
    return ChatPromptTemplate.from_messages(
        [(msg['role'], msg['content']) for msg in messages]
    )
def load_information_from_json(settings:Settings,model_embedding:Sentences_Transformer_Embedding):
    current_dir=os.path.dirname(__file__)
    json_path=os.path.join(current_dir,'..','core',settings.PATH_INFOR)
    json_path=os.path.abspath(json_path)
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    corpus=[item['content'] for item in data]
    corpus_embedding=model_embedding.embeddings_bkai.embed_documents(corpus)
    return corpus,corpus_embedding
def search_from_json(corpus_embedding,corpus,query,model_embedding:Sentences_Transformer_Embedding):
    query_embedding=model_embedding.embeddings_bkai.embed_query(query)
    cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
    top_results = cos_scores.argsort(descending=True)[:5]
    results=[]
    for idx in top_results:
        results.append(corpus[idx])
    return "\n".join(results)

def clean_generated_queries(queries):
        cleaned_queries = []
        for query in queries:
            if '```' in query:
                continue
            if len(query.split()) < 5:
                continue
            cleaned_queries.append(query)
        return cleaned_queries
def extract_json_dict(text):
    if text is None:
        return {}
    text = text.strip()
    # Case 1: fenced ```json {...} ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Case 2: bare JSON object anywhere in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    # Case 3: no JSON at all — treat as "no entities detected"
    return {}
    
def clean_code_fence_safe(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        # Nếu dòng đầu chỉ chứa dấu ``` hoặc ```json
        if lines[0].strip() == "```" or lines[0].strip().startswith("```"):
            lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def parse_raw_json(raw_text: str) -> dict:
    """Extract {"answer": ..., "key": [...]} from a JSON-ish string returned
    by the generator.

    Handles three cases:
    - Well-formed JSON → json.loads directly.
    - JSON with an `answer` string that contains unescaped quotes/newlines
      (Claude sometimes does this) → apply the targeted regex fix.
    - Empty answer string → allowed.
    """
    if raw_text is None:
        raise ValueError("empty generator output")
    text = raw_text.replace('“', '"').replace('”', '"').strip()

    # Fast path — try strict JSON first; Claude is usually well-behaved.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj.setdefault("answer", "")
            obj.setdefault("key", [])
            return obj
    except json.JSONDecodeError:
        pass

    # Slow path — regex-repair an `answer` string that contains raw quotes.
    # `.*?` (not `.+?`) so empty answers match too.
    pattern = r'("answer"\s*:\s*")(.*?)"(?=\s*,\s*"key"|\s*})'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        print("DEBUG - raw_text:\n", raw_text)
        raise ValueError("Không tìm thấy trường 'answer' hoặc định dạng quá lệch không parse được.")

    prefix = match.group(1)
    raw_content = match.group(2)
    end_pos = match.end()

    escaped_content = json.dumps(raw_content)[1:-1]
    fixed_text = (
        text[: match.start(1)]
        + prefix
        + escaped_content
        + '"'
        + text[end_pos:]
    )
    fixed_text = textwrap.dedent(fixed_text).strip()
    return json.loads(fixed_text)



