import os, sys, json, time, re, argparse, asyncio, hashlib
import requests
from google import genai
from google.genai import types

COOLDOWN_SEARCH = 10
COOLDOWN_DEEP = 60

def get_args():
    parser = argparse.ArgumentParser(description="Gemini Free Tier Researcher")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model name")
    parser.add_argument("--mode", choices=["search", "deep"], default="search")
    parser.add_argument("--template", required=True, help="Path to template.txt")
    parser.add_argument("--values", help="Path to values.json")
    parser.add_argument("--files", help="Comma-separated paths to local files (PDF, TXT, etc.)")
    return parser.parse_args()

def get_file_hash(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def upload_with_cache(client, path):
    file_hash = get_file_hash(path)
    display_name = f"hash_{file_hash}"
    for f in client.files.list():
        if f.display_name == display_name:
            print(f"   - Cache hit: {os.path.basename(path)} is already in cloud.")
            return f
    print(f"   - Uploading new file: {os.path.basename(path)}...")
    uploaded_file = client.files.upload(file=path, config={'display_name': display_name})
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    return uploaded_file

async def get_final_url(redirect_url):
    try:
        res = requests.head(redirect_url, allow_redirects=True, timeout=5)
        return res.url
    except Exception:
        return redirect_url

async def run_research(client, model_id, mode, prompt, files):
    content_parts = files + [prompt]
    if mode == "search":
        search_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[search_tool],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF")
            ]
        )
        res = await client.aio.models.generate_content(
            model=model_id, 
            contents=content_parts, 
            config=config
        )
        output_text = res.text
        if not output_text:
            return "ERROR: Empty response from model (Safety Block?)"
        try:
            metadata = res.candidates[0].grounding_metadata
            if metadata and metadata.grounding_chunks:
                for i, chunk in enumerate(metadata.grounding_chunks):
                    if chunk.web:
                        clean_url = await get_final_url(chunk.web.uri)
                        placeholder = f"[Source {i+1}]"
                        output_text = output_text.replace(placeholder, clean_url)
                        output_text = output_text.replace(chunk.web.uri, clean_url)
            else:
                print(f"      (Note: No live search results used for this paragraph.)")
        except Exception as e:
            print(f"      (Link cleaning failed: {e})")
        return output_text
    else:
        interaction = client.interactions.create(
            model=model_id,
            input=content_parts,
            agent_id="deep-research",
            background=True
        )
        while True:
            status = client.interactions.get(id=interaction.id)
            if status.state == "COMPLETED":
                return status.result.text
            elif status.state == "FAILED":
                raise Exception(f"Deep Research failed: {status.error}")
            await asyncio.sleep(10)

async def main():
    args = get_args()
    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("Error: GEMINI_API_KEY missing.")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    context_files = []
    if args.files:
        print("Checking/Uploading context files...")
        for p in args.files.split(','):
            context_files.append(upload_with_cache(client, p.strip()))
    with open(args.template, 'r') as f: template = f.read()
    values = json.load(open(args.values, 'r')) if args.values else [{}]
    os.makedirs("outputs", exist_ok=True)
    try:
        for i, entry in enumerate(values, start=1):
            prompt = template.format(**entry)
            val = list(entry.values())[0] if entry else "output"
            name = re.sub(r'[^\w\s-]', '', str(val)).strip().lower().replace(' ', '_')
            print(f"[{i}/{len(values)}] Mode: {args.mode.upper()} | Target: {val}")
            try:
                output = await run_research(client, f"models/{args.model}", args.mode, prompt, context_files)
                with open(f"outputs/{i:03d}_{name}.txt", "w") as f:
                    f.write(output)
                if i < len(values):
                    wait_time = COOLDOWN_DEEP if args.mode == "deep" else COOLDOWN_SEARCH
                    print(f"   Success. Cooldown for {wait_time}s...")
                    await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"   !! Error on {val}: {e}")
    finally:
        if context_files:
            print("\nCleaning up cloud storage...")
            for f in context_files:
                try:
                    client.files.delete(name=f.name)
                    print(f"   - Deleted: {f.display_name}")
                except:
                    pass

if __name__ == "__main__":
    asyncio.run(main())
