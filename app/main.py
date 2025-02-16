#########################
# main.py
#########################

import csv
import json
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import requests
from dateutil.parser import parse
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import traceback

load_dotenv()

#######################################################################
# GLOBALS & CONFIG
#######################################################################
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN") or "YOUR_TOKEN_HERE"
AIPROXY_CHAT_URL = os.getenv("AIPROXY_CHAT_URL", "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions")
MODEL_NAME = "gpt-4o-mini"

DATA_DIR = Path("/data")
# DATA_DIR = Path("/Users/parmeet/pycharm/tds-proj-1/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)


#######################################################################
# SECURITY & PATH HELPERS (B1, B2 enforcement)
#######################################################################
# B1: Data outside /data is never accessed
# B2: Data is never deleted
# => We'll never do "rm" or allow paths escaping /data.



def safe_resolve(user_path: str) -> Path:
    """Safely resolve user_path within DATA_DIR, or raise HTTPException."""
    # user_path = user_path.lstrip("/")  # remove leading slash if present
    if not is_safe_path(user_path):
        raise HTTPException(status_code=403, detail=f"Path escapes /data: {user_path}")
    return Path(user_path)


#######################################################################
# TASK IMPLEMENTATIONS (A1..A10, B3..B10)
#######################################################################
def install_and_run_script_if_needed(package: str, script_url: str, script_arg: str):
    """
    A1. Install a specified software package (if necessary) and execute a provided script 
    with a given argument to generate required data files.
    """
    # 1) pip install <package>
    subprocess.run(["python3", "-m", "pip", "install", package], check=True)

    # 2) Download the script
    local_name = script_url.split("/")[-1]
    subprocess.run(["curl", "-o", local_name, script_url], check=True)

    # 3) Run the script with the given argument
    # Example: python3 downloaded_script.py <script_arg>
    cmd = ["python3", local_name, script_arg]
    subprocess.run(cmd, check=True)


def format_file_inplace(file_path: str, prettier_version: str):
    """
    A2. Format the contents of a specified file using Prettier in-place.
    """
    target = safe_resolve(file_path)
    # Must have "npm install -g prettier@<version>" in Dockerfile
    cmd = ["npx", f"prettier@{prettier_version}", "--write", str(target)]
    subprocess.run(cmd, check=True)


def count_specific_weekday(input_file: str, output_file: str, weekday: str):
    """
    A3. Analyze a list of dates from input_file, count occurrences of `weekday` (0=Mon,1=Tue,2=Wed,...),
    write the count to output_file.
    """
    weekday = int(weekday)
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)

    lines = inp.read_text().splitlines()
    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            dt = parse(line)
            print(f"Parsed date: {line} -> {dt}, Weekday: {dt.weekday()}") # Debug print
            if dt.weekday() == weekday:
                print(f"  - Matched weekday {weekday}!")
                count += 1
        except:
            print(f"Error parsing line: {line}, Error: {e}")  # Debug print
            pass  # ignore parse errors
    out.write_text(str(count))
    print(f"Final count: {count}")


def sort_contacts_by_fields(input_file: str, output_file: str, sort_fields: str):
    """
    A4. Sort a JSON array of contacts by the given fields in order (e.g. ["last_name","first_name"]).
    """
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)
    try:
        data = json.loads(inp.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Input file not valid JSON: {e}")

    def sort_func(item):
        return tuple(item.get(k, "") for k in sort_fields.split(','))

    data.sort(key=sort_func)
    out.write_text(json.dumps(data, indent=2))


def retrieve_log_lines(input_dir: str, output_file: str, lines_per_file: str, max_logs: str):
    """
    A5. Retrieve the specified number of lines_per_file from each of the most recent log files
    in input_dir (by modification time), and write them to output_file (most recent first).
    """
    lines_per_file = int(lines_per_file)
    max_logs = int(max_logs)
    d = safe_resolve(input_dir)
    out = safe_resolve(output_file)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Input dir not found or not a directory")

    # list .log files sorted by mtime desc
    log_files = [f for f in d.iterdir() if f.is_file() and f.suffix == ".log"]
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    log_files = log_files[:max_logs]

    all_lines = []
    for logf in log_files:
        lines = logf.read_text().splitlines()
        snippet = lines[:lines_per_file]
        all_lines.extend(snippet)

    out.write_text("\n".join(all_lines) + "\n")


def index_files_by_extension(input_dir: str, output_file: str, extension: str, marker: str):
    """
    A6. Identify all files with `extension` in `input_dir`. For each file, extract the first line 
    starting with `marker`, create an index mapping {relative_filename: extracted_text}, 
    and save as JSON to `output_file`.
    """
    d = safe_resolve(input_dir)
    out = safe_resolve(output_file)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail="Input dir not found")

    index_map = {}
    for f in d.rglob(f"*{extension}"):
        rel_path = f.relative_to(d)
        text = f.read_text().splitlines()
        found = ""
        for line in text:
            if line.startswith(marker):
                found = line[len(marker):].strip()
                break
        index_map[str(rel_path)] = found

    out.write_text(json.dumps(index_map, indent=2))


def extract_info_with_llm(input_file: str, output_file: str, prompt_text: str):
    """
    A7. Extract specific info (e.g. sender's email) from the content of a text file 
    using an LLM, write the extracted info to output_file.
    """
    # We'll do a minimal LLM call (no function calling).
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)
    content = inp.read_text()

    # Send to LLM
    msg = [
        {"role": "system", "content": "You are an assistant that extracts requested info from text."
                                      "Only provide the requested info, nothing else."},
        {"role": "user", "content": f"{prompt_text}\n\nText:\n{content}"}
    ]
    extracted = call_simple_llm(msg)
    out.write_text(extracted.strip())


def extract_text_from_image(image_path: str, output_file: str, prompt_text: str):
    """
    A8. Process an image (e.g. credit card). Could use Tesseract or pass to LLM as base64.
    We'll demonstrate a Tesseract approach + minimal text post-processing.
    """
    img = safe_resolve(image_path)
    out = safe_resolve(output_file)

    # 1) OCR with Tesseract
    cmd = ["tesseract", img.name, "stdout"]
    proc = subprocess.run(cmd, cwd=img.parent, capture_output=True, text=True, check=True)
    raw = proc.stdout

    # 2) Optionally pass raw to LLM with `prompt_text` if you want advanced parsing
    # For now, let's just remove spaces
    cleaned = raw.replace(" ", "").strip()

    out.write_text(cleaned)
    full_path_str = str(out)

    extract_info_with_llm(full_path_str, full_path_str, cleaned)


def find_most_similar_pair(input_file: str, output_file: str):
    """
    A9. Analyze a list of textual lines, find the most similar pair. 
    We'll do a naive approach with embeddings from the AI Proxy if available.
    """
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)
    lines = [l.strip() for l in inp.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        out.write_text("")
        return

    # call embeddings
    embeddings = get_embeddings(lines)
    # compute similarity
    sim = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sim, -1)
    i, j = np.unravel_index(sim.argmax(), sim.shape)
    pair = sorted([lines[i], lines[j]])
    out.write_text("\n".join(pair) + "\n")


def query_db_and_write(db_file: str, output_file: str, sql_query: str):
    """
    A10. Query a database (assumed sqlite) for a specific metric, write result to output_file.
    """
    dbp = safe_resolve(db_file)
    out = safe_resolve(output_file)
    conn = sqlite3.connect(dbp)
    try:
        c = conn.cursor()
        c.execute(sql_query)
        val = c.fetchone()
        if val is None:
            val = (0,)
        result = val[0]
    finally:
        conn.close()

    out.write_text(str(result))


##################################################################
# B3. Fetch data from an API and save it
##################################################################
def fetch_data_from_api(url: str, output_file: str, method: str = "GET", payload: Optional[dict] = None):
    """
    B3. Example: fetch data from an API (GET or POST) and write JSON to output_file. 
    """
    out = safe_resolve(output_file)
    if method.upper() == "GET":
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        out.write_text(json.dumps(data, indent=2))
    else:
        # POST
        r = requests.post(url, json=payload or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        out.write_text(json.dumps(data, indent=2))


##################################################################
# B4. Clone a git repo and make a commit
##################################################################
def clone_repo_and_commit(repo_url: str, output_dir: str, commit_msg: str):
    """
    B4. Clone a repo into /data/<output_dir>, make a commit with commit_msg.
    """
    target = safe_resolve(output_dir)
    # 1) git clone
    subprocess.run(["git", "clone", repo_url, str(target)], check=True)
    # 2) make some change if needed, or just commit
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=target, check=True)


##################################################################
# B5. Run a SQL query on SQLite or DuckDB
##################################################################
def run_sql_on_db(db_file: str, output_file: str, sql_query: str, db_type: str = "sqlite"):
    """
    B5. Run a SQL query on either SQLite or DuckDB, write results to output_file.
    """
    dbp = safe_resolve(db_file)
    outp = safe_resolve(output_file)

    if db_type.lower() == "sqlite":
        conn = sqlite3.connect(dbp)
    else:
        conn = duckdb.connect(str(dbp))

    try:
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        # write them out
        with outp.open("w") as f:
            for row in rows:
                f.write(str(row) + "\n")
    finally:
        conn.close()


##################################################################
# B6. Extract data (scrape) from a website
##################################################################
def scrape_website(url: str, output_file: str):
    """
    B6. Download HTML from a site, write to output_file.
    """
    out = safe_resolve(output_file)
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    html = r.text
    out.write_text(html)


##################################################################
# B7. Compress or resize an image
##################################################################
def compress_or_resize_image(input_file: str, output_file: str, width: str, height: str, quality: str = '75'):
    """
    B7. Example uses Pillow (pip install pillow). Resize to (width,height) and save with 'quality'.
    """
    width = int(width)
    height = int(height)
    quality = int(quality)
    from PIL import Image
    inp = safe_resolve(input_file)
    outp = safe_resolve(output_file)

    im = Image.open(inp)
    im = im.resize((width, height))
    im.save(outp, optimize=True, quality=quality)


##################################################################
# B8. Transcribe audio from MP3
##################################################################
def transcribe_audio(input_file: str, output_file: str):
    """
    B8. Stub for audio transcription. Could call e.g. Whisper or any speech-to-text engine.
    We'll just write a placeholder.
    """
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)
    # Real code might call an external ASR engine
    # We'll store "Transcribed text" as a placeholder
    out.write_text("Transcribed text (placeholder)")


##################################################################
# B9. Convert Markdown to HTML
##################################################################
def convert_md_to_html(input_file: str, output_file: str):
    """
    B9. Convert .md to HTML with Python markdown or similar.
    """
    import markdown
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)
    md_text = inp.read_text()
    html = markdown.markdown(md_text)
    out.write_text(html)


##################################################################
# B10. Filter CSV file and return JSON data
# (Here we simply write results to a file, or you can serve them as an endpoint.)
##################################################################
def filter_csv_and_write_json(input_file: str, output_file: str, column: str, value: str):
    """
    B10. Filter a CSV by <column>=<value>, write results as JSON.
    """
    inp = safe_resolve(input_file)
    out = safe_resolve(output_file)

    rows = []
    with inp.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get(column) == value:
                rows.append(row)

    out.write_text(json.dumps(rows, indent=2))


#######################################################################
# HELPER: LLM CALLS
#######################################################################
def call_simple_llm(messages: List[Dict[str, Any]]) -> str:
    """
    Minimal helper to call GPT-4o-Mini with a small prompt, returning text.
    No function calling here, just raw completion.
    """
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.2
    }
    try:
        resp = requests.post(AIPROXY_CHAT_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return text
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")


def get_embeddings(lines: List[str]) -> np.ndarray:
    """
    Example embedding call with AI Proxy:
    POST to /v1/embeddings => model="text-embedding-3-small"
    """
    # You might set a different endpoint. 
    EMBED_URL = AIPROXY_CHAT_URL.replace("/chat/completions", "/embeddings")
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": lines}
    resp = requests.post(EMBED_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    emb_json = resp.json()
    # parse out embeddings
    arr = [entry["embedding"] for entry in emb_json["data"]]
    return np.array(arr)


#######################################################################
# OPENAI FUNCTION-CALLING SETUP
#######################################################################
# We'll define a dictionary of the above tasks by name => function
TASK_MAP = {
    # A1..A10
    "install_and_run_script_if_needed": install_and_run_script_if_needed,
    "format_file_inplace": format_file_inplace,
    "count_specific_weekday": count_specific_weekday,
    "sort_contacts_by_fields": sort_contacts_by_fields,
    "retrieve_log_lines": retrieve_log_lines,
    "index_files_by_extension": index_files_by_extension,
    "extract_info_with_llm": extract_info_with_llm,
    "extract_text_from_image": extract_text_from_image,
    "find_most_similar_pair": find_most_similar_pair,
    "query_db_and_write": query_db_and_write,

    # B3..B10
    "fetch_data_from_api": fetch_data_from_api,
    "clone_repo_and_commit": clone_repo_and_commit,
    "run_sql_on_db": run_sql_on_db,
    "scrape_website": scrape_website,
    "compress_or_resize_image": compress_or_resize_image,
    "transcribe_audio": transcribe_audio,
    "convert_md_to_html": convert_md_to_html,
    "filter_csv_and_write_json": filter_csv_and_write_json,
}

#######################################################################
# SCHEMA GENERATION (Optional)
#######################################################################
import inspect
from docstring_parser import parse as docparse
from pydantic import create_model


def function_to_schema(func):
    """Convert a Python function signature + docstring into an OpenAI 'function' schema."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    parsed = docparse(doc)

    # Build a pydantic model from parameters
    fields = {}
    for pname, param in sig.parameters.items():
        # For brevity, assume everything is string except we detect List[str] or int
        # You can do more sophisticated type introspection if you want.
        fields[pname] = (str, ...)
    PModel = create_model(func.__name__ + "Model", **fields)
    schema = PModel.model_json_schema()
    props = schema["properties"]

    # Inject docstring param descriptions
    param_docs = {p.arg_name: p.description for p in parsed.params}
    for pr in props:
        if pr in param_docs:
            props[pr]["description"] = param_docs[pr]

    # short desc from docstring:
    short_desc = parsed.short_description or ""

    return {
        "name": func.__name__,
        "description": short_desc,
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(props.keys())
        }
    }


def get_all_function_schemas():
    schemas = []
    for name, fn in TASK_MAP.items():
        schemas.append(function_to_schema(fn))
    return schemas


#######################################################################
# CALL LLM with function calling
#######################################################################
def call_llm_function_call(task_description: str) -> dict:
    """
    Sends the user task + function schemas to GPT-4o-Mini, letting it pick a function + arguments.
    """
    tools = get_all_function_schemas()
    headers = {"Authorization": f"Bearer {AIPROXY_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an automation agent. You have these functions available."},
            {"role": "user", "content": task_description},
        ],
        "functions": tools,
        "function_call": "auto",
        "max_tokens": 200,
        "temperature": 0.2
    }
    resp = requests.post(AIPROXY_CHAT_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def execute_llm_function_call(llm_response: dict):
    """
    Parse the LLM's chosen function + arguments and call the appropriate Python function.
    """
    choices = llm_response.get("choices", [])
    if not choices:
        raise ValueError("No choices in LLM response.")

    msg = choices[0].get("message", {})
    fn_call = msg.get("function_call")
    if not fn_call:
        raise ValueError("LLM did not pick a function.")

    fn_name = fn_call["name"]
    raw_args = fn_call["arguments"]
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function arguments: {raw_args}")

    func = TASK_MAP.get(fn_name)
    print(f"Executing function {fn_name} with args={args}")
    if not func:
        raise ValueError(f"No function named {fn_name} is defined.")

    # Call the function with the arguments
    try:
        logging.info(f"Running {fn_name} with args={args}")
        func(**args)
    except TypeError as e:
        raise ValueError(f"Argument mismatch: {e}")
    except Exception as e:
        # Raise as HTTPException so the user sees the error
        raise HTTPException(status_code=500, detail=f"Error in {fn_name}: {e}")


#######################################################################
# FASTAPI ROUTES
#######################################################################
@app.post("/run")
def run_task(task: str = Query(..., description="Plain-English task description.")):
    """
    The main entrypoint. We interpret 'task' with the LLM function calling approach.
    Then we execute the chosen function (e.g. installing, formatting, sorting, etc.).
    """
    try:
        llm_response = call_llm_function_call(task)
        execute_llm_function_call(llm_response)
        return {"status": "success", "message": "Task executed successfully"}
    except HTTPException as http_exc:
        # re-raise
        raise http_exc
    except Exception as e:
        tb = "".join(reversed(traceback.format_exc()))
        raise HTTPException(status_code=500, detail=str(e), headers={"X-Traceback": tb})


@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="The file path to read, including the /data/ prefix.")):
    """Reads a file and returns its content.

    Args:
        path: The *full* path to the file, *including* the /data/ prefix.
             e.g., /data/myfile.txt, /data/subdir/anotherfile.txt
    """
    if not path.startswith("/data/"):
        raise HTTPException(status_code=403, detail="Access denied. Path must start with /data/")

    # Use os.path.join for safe path construction
    absolute_path = os.path.join("/", path)  # Join with root to make it absolute

    if not os.path.exists(absolute_path) or not os.path.isfile(absolute_path):
        raise HTTPException(status_code=404, detail="")

    # is_safe_path check using os.path
    if not is_safe_path(path):  # Pass the relative path to is_safe_path
        raise HTTPException(status_code=403, detail="Access denied. Path must be within the data directory.")

    try:
        with open(absolute_path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")


def is_safe_path(path: str) -> bool:
    """Validates the file path to prevent directory traversal."""
    try:
        # DATA_DIR should be an absolute path
        absolute_data_dir = DATA_DIR.resolve()  # Make sure it's absolute

        # Construct the absolute path to the requested file
        absolute_requested_path = Path(os.path.join("/", path)).resolve()

        # Check if the resolved requested path is within DATA_DIR
        return absolute_requested_path.is_relative_to(absolute_data_dir)
    except Exception:
        return False


# If running locally, uncomment:
if __name__ == "__main__":
    import uvicorn, traceback
    uvicorn.run(app, host="0.0.0.0", port=8000)
