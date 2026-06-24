import json
import argparse
import requests
import ida_funcs
import ida_kernwin
import ida_name
import idaapi

LLM_MODEL = "qwen2.5-coder:1.5b"
LLM_HOST = "http://100.104.79.110:11434"  # tailscale
REPEATABLE = True   # True => also shows at call sites; False => only at the function

def query_llm(prompt):
    """Query the LLM via HTTP requests"""
    try:
        url = f"{LLM_HOST}/api/generate"
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[summaries] Error querying LLM: {e}")
        return None


def wrap_text(text, width=80):
    """Wrap text to width with newlines, preserving existing paragraph breaks"""
    paragraphs = text.split('\n')
    wrapped_paragraphs = []
    
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_paragraphs.append('')
            continue
        
        lines = []
        current_line = []
        
        for word in words:
            if sum(len(w) for w in current_line) + len(current_line) + len(word) <= width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        wrapped_paragraphs.append("\n".join(lines))
    
    return "\n".join(wrapped_paragraphs)


def annotate_summaries(path, repeatable=REPEATABLE, save=False):
    """Annotate the open database with summaries from a JSON file.

    Call from the IDA Python console, e.g.:
        annotate_summaries("path_to_summaries.json")

    Args:
        path: Path to summaries JSON file
        repeatable: True => show at call sites; False => only at function definition
        save: Save database after annotation

    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.items()
    else:
        records = ((e["name"], e.get("summary", e.get("comment", "")))
                   for e in data)

    applied = skipped = missing = 0
    
    for name, summary in records:
        if not summary:
            skipped += 1
            continue

        ea = ida_name.get_name_ea(idaapi.BADADDR, name)
        if ea == idaapi.BADADDR:
            print(f"[summaries] name not found: {name}")
            missing += 1
            continue

        func = ida_funcs.get_func(ea)
        if func is None:
            print(f"[summaries] not a function: {name} @ {ea:#x}")
            missing += 1
            continue

        # Set full summary at function definition
        wrapped_summary = "\n" + wrap_text(summary)
        if ida_funcs.set_func_cmt(func, wrapped_summary, repeatable):
            applied += 1
        else:
            skipped += 1

    print(f"[summaries] applied={applied} skipped={skipped} missing={missing}")

    if not ida_kernwin.cvar.batch:
        ida_kernwin.refresh_idaview_anyway()

    if save:
        idc_save = idaapi.save_database
        idc_save("")   # "" => save to the current database path
        print("[summaries] database saved")
