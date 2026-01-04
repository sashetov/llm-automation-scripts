import os
import sys
import argparse
import asyncio
import requests
import re
import time
from pathlib import Path
from google import genai
from google.genai import types

COOLDOWN_SEARCH = 5
MAX_ATTEMPTS = 10

def get_args():
    parser = argparse.ArgumentParser(description="Gemini Multi-Step Fact-Checking Auditor")
    parser.add_argument("--query", required=True, help="Path to file containing <INIT_QUERY>")
    parser.add_argument("--context", required=True, help="Path to file containing <INIT_CONTEXT>")
    parser.add_argument("--files", help="Comma-separated paths to additional context files")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--outdir", default="output", help="Directory to save responses")
    parser.add_argument("--min_urls", type=int, default=None, help="Optional: Min verified URLs for Step 0 and 5")
    return parser.parse_args()

def read_file(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {path}: {e}")
        sys.exit(1)

def extract_urls(text):
    return re.findall(r'https?://[^\s)\]]+', text)

async def get_final_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        clean_url = url.strip(').,[]')
        res = requests.head(clean_url, headers=headers, allow_redirects=True, timeout=5)
        return res.url, (res.status_code < 500)
    except:
        return url, False

async def verify_response_links(response_obj):
    output_text = response_obj.text
    live_urls = set()
    dead_links = []

    candidate = response_obj.candidates[0]
    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata.grounding_chunks:
        for chunk in candidate.grounding_metadata.grounding_chunks:
            if chunk.web and chunk.web.uri:
                live_urls.add(chunk.web.uri)

    found_in_text = extract_urls(output_text)
    all_found = live_urls.union(set(found_in_text))
    for url in all_found:
        f_url, is_live = await get_final_url(url)
        if is_live: live_urls.add(f_url)
        else: dead_links.append(f_url)

    return output_text, dead_links, list(live_urls)

async def verified_generate(target, config, prompt, model_id, is_chat=False, contents=None, min_urls=None):
    attempts = 0
    working_target = target
    if not is_chat and min_urls:
        working_target = target.chats.create(model=model_id, config=config)
        is_chat = True
    while attempts < MAX_ATTEMPTS:
        call_prompt = prompt if attempts == 0 else f"SEARCH THE WEB to find sources. {prompt}"
        if is_chat:
            res = working_target.send_message(call_prompt)
        else:
            res = target.models.generate_content(model=model_id, contents=contents or [call_prompt], config=config)
        text, dead, live = await verify_response_links(res)
        print(f"   - Attempt {attempts + 1}: Found {len(live)} live, {len(dead)} dead.")
        needs_more = (min_urls is not None and len(live) < min_urls)
        if not dead and not needs_more:
            return text
        attempts += 1
        prompt = f"I need more sources. Use Google Search to find at least {min_urls} live URLs. Previous attempt found {len(live)}."
        await asyncio.sleep(COOLDOWN_SEARCH)
    return text

async def upload_files(client, file_paths):
    uploaded = []
    if not file_paths: return uploaded
    for path in file_paths.split(','):
        path = path.strip()
        print(f"   - Uploading {path}...")
        f = client.files.upload(path=path)
        while f.state.name == "PROCESSING":
            await asyncio.sleep(2)
            f = client.files.get(name=f.name)
        uploaded.append(f)
    return uploaded

async def run_chain():
    search_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[search_tool],
        safety_settings=[
            types.SafetySetting( category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting( category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting( category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting( category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting( category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
        ]
    )
    args = get_args()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    out_path = Path(args.outdir); out_path.mkdir(exist_ok=True)
    init_query = read_file(args.query)
    init_context = read_file(args.context)
    uploaded_files = await upload_files(client, args.files)
    print(f"--- Step 0: Initial Response (Cooldown {COOLDOWN_SEARCH}s) ---")
    prompt0 = f"Context: {init_context}\n\nQuestion: {init_query}"
    resp0_text = await verified_generate(client, config, prompt0, args.model, contents=uploaded_files + [prompt0], min_urls=args.min_urls)
    (out_path / "GEN_RESP0.txt").write_text(resp0_text)
    print("--- Step 1: Claim Extraction (Cooldown 5s) ---")
    prompt1 = f"Extract all factual claims from this text into a numbered list:\n{resp0_text}"
    gen_resp1 = await verified_generate(client, config, prompt1, args.model, contents=[prompt1])
    (out_path / "GEN_RESP1.txt").write_text(gen_resp1)
    print("--- Step 2: Search Verification (Cooldown 5s) ---")
    chat_search = client.chats.create(model=args.model, config=config)
    prompt2 = f"""Verify each claim individually
- provide a citation or URL for the answer, if possible
- questions:
{gen_resp1}"""
    gen_resp2 = await verified_generate(chat_search, config, prompt2, args.model, contents=uploaded_files + [prompt0], is_chat=True)
    (out_path / "GEN_RESP2.txt").write_text(gen_resp2)
    print("--- Step 3: Synthesis (Cooldown 5s) ---")
    prompt3 = f"Using only verified answers you provided above, please answer: {init_query}"
    resp3 = chat_search.send_message(prompt3)
    gen_resp3 = resp3.text
    (out_path / "GEN_RESP3.txt").write_text(gen_resp3)
    await asyncio.sleep(COOLDOWN_SEARCH)
    print("--- Step 4: Independent Audit (Cooldown 5s) ---")
    prompt4 = f"""Role: you are an independent auditor evaluating an AI-generated response.
Evaluate only.

Input:
Original question: {init_query}
Original context: {init_context}

Response that you need to audit:
{gen_resp3}

Your task:
Fact-check this response. 
If you are unsure or the information is missing, say "I don't know" instead of guessing.
For each main claim, add a confidence interval in parentheses (high), (medium) or (low).
At the end, list anything you are unsure about or could not find."""
    gen_resp4 = await verified_generate(client, config, prompt2, args.model, contents=uploaded_files + [prompt0])
    (out_path / "GEN_RESP4.txt").write_text(gen_resp4)
    print("--- Step 5: Final Synthesis (Verified Loop) ---")
    instruction = """(re)write in this format:
- single tight paragraph (only produce one paragraph as an output)
- dont use em dash
- speak plain english, no jargon
- don't repeat yourself
- retain/correct/enrich any existing tables
- be succinct, no filler fluff wording while also detailed, data oriented and logical
- source EVERY claim in parentheses with real non-hallucinated source URLS
- include large quotes from any source you use
- structure the paragraph as a linear timeline narrative moving forward in time and citing exact dates/times
- and at last, to help me confirm you are not hallucinating:
  - make a table with the quote and the URL link (from your google search tool and or grounding api) that I can click to confirm its real
  - add name/url to sources you cited that were not quoted from in same table
  - put table after the single paragraph you produced
"""
    final_prompt = f"{instruction}\n\nAudit Results for context:\n{gen_resp4}\n\nOriginal Query: {init_query}"
    final_text = await verified_generate(client, config, final_prompt, args.model, min_urls=args.min_urls)
    (out_path / "GEN_FINAL.txt").write_text(final_text)
    print(f"\nSuccess! Final output saved to {args.outdir}/GEN_FINAL.txt")

if __name__ == "__main__":
    asyncio.run(run_chain())
