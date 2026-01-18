#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict, Iterable, List, Optional


def _extract_text(content: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if isinstance(parts, list):
        out: List[str] = []
        for part in parts:
            if part is None:
                continue
            if isinstance(part, str):
                out.append(part)
            else:
                out.append(json.dumps(part, ensure_ascii=False))
        if out:
            return "\n".join(out)
        return None
    text = content.get("text")
    if isinstance(text, str):
        return text
    return None


def _iter_nodes_path(mapping: Dict[str, Any], current_node: Optional[str]) -> List[Dict[str, Any]]:
    if not current_node or current_node not in mapping:
        return []
    chain: List[Dict[str, Any]] = []
    seen = set()
    node_id = current_node
    while node_id and node_id not in seen:
        seen.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            break
        chain.append(node)
        node_id = node.get("parent")
    chain.reverse()
    return chain


def _iter_nodes_all(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    roots = [node_id for node_id, node in mapping.items() if isinstance(node, dict) and node.get("parent") is None]
    ordered: List[Dict[str, Any]] = []
    visited = set()

    def dfs(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            return
        ordered.append(node)
        for child_id in node.get("children", []):
            if isinstance(child_id, str):
                dfs(child_id)

    for root_id in roots:
        dfs(root_id)
    if not roots:
        for node_id in mapping.keys():
            if isinstance(node_id, str):
                dfs(node_id)
    return ordered


def _iter_nodes(mapping: Dict[str, Any], current_node: Optional[str], include_all: bool) -> Iterable[Dict[str, Any]]:
    if include_all:
        return _iter_nodes_all(mapping)
    path_nodes = _iter_nodes_path(mapping, current_node)
    return path_nodes if path_nodes else _iter_nodes_all(mapping)


def _write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text(records: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            idx = record.get("index")
            f.write(f"========== MESSAGE {idx} ==========\n")
            f.write(f"role: {record.get('role')}\n")
            name = record.get("name") or ""
            f.write(f"name: {name}\n")
            f.write(f"create_time: {record.get('create_time')}\n")
            f.write(f"content_type: {record.get('content_type')}\n")
            f.write(f"message_id: {record.get('message_id')}\n")
            content = record.get("content") or ""
            f.write(f"content_length: {len(content)}\n")
            f.write("CONTENT_START\n")
            if content:
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
            f.write("CONTENT_END\n")
            f.write(f"========== END MESSAGE {idx} ==========\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract visible chat content from withgpt.json.")
    parser.add_argument("--input", default=".cache/withgpt.json", help="Path to withgpt.json")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text", help="Output format")
    parser.add_argument("--all", action="store_true", help="Include all nodes instead of only the current path")
    parser.add_argument("--include-hidden", action="store_true", help="Include visually hidden messages")
    parser.add_argument("--include-system", action="store_true", help="Include system role messages")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = data.get("mapping", {})
    if not isinstance(mapping, dict):
        raise SystemExit("Invalid JSON: mapping is not an object")

    current_node = data.get("current_node")
    nodes = _iter_nodes(mapping, current_node, args.all)

    records: List[Dict[str, Any]] = []
    index = 0
    for node in nodes:
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        metadata = msg.get("metadata", {})
        if not args.include_hidden and isinstance(metadata, dict):
            if metadata.get("is_visually_hidden_from_conversation"):
                continue
        author = msg.get("author", {})
        role = author.get("role") if isinstance(author, dict) else None
        if not role:
            continue
        if role == "system" and not args.include_system:
            continue
        content = msg.get("content")
        text = _extract_text(content)
        if not text:
            continue

        index += 1
        records.append(
            {
                "index": index,
                "role": role,
                "name": author.get("name") if isinstance(author, dict) else None,
                "create_time": msg.get("create_time"),
                "content_type": content.get("content_type") if isinstance(content, dict) else None,
                "content": text,
                "message_id": msg.get("id"),
            }
        )

    output_path = args.output
    if not output_path:
        output_path = "withgpt_extracted.txt" if args.format == "text" else "withgpt_extracted.jsonl"

    if args.format == "jsonl":
        _write_jsonl(records, output_path)
    else:
        _write_text(records, output_path)

    print(f"Wrote {len(records)} messages to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
