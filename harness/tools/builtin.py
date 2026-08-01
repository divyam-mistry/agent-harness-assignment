"""Built-in tools backed by local sample data (for run_chat.py and local testing)."""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import random
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from rank_bm25 import BM25Okapi

from harness.api import ToolResult, ToolSpec

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

DOCS: dict[str, tuple[str, str]] = {
    "kb://policies/shipping": ("Shipping policy", "Standard shipping takes 3-5 business days within the EU and 6-10 days elsewhere. Express shipping takes 1-2 business days and costs 14.90. Orders can be redirected to a different address until they are marked as dispatched. Deliveries to office addresses require a named recipient and a floor number. We do not ship to PO boxes for items over 2 kg."),
    "kb://policies/returns": ("Returns policy", "Items can be returned within 30 days of delivery in their original condition. Refunds are issued to the original payment method within 5 business days of the return arriving at the warehouse. Sale items are final sale unless faulty. Return labels are free for members of the Plus programme."),
    "kb://policies/plus": ("Plus membership", "Plus costs 49.00 per year. Members get free express shipping, free return labels, and priority support. Plus can be cancelled within 14 days for a full refund."),
    "kb://policies/price-match": ("Price match", "We match the price of identical items sold by authorised retailers within 14 days of purchase. The item must be in stock at the competitor. Marketplace sellers are excluded."),
    "kb://policies/warranty": ("Warranty", "Electronics carry a 24-month warranty. Outdoor gear carries a 12-month warranty against manufacturing defects. Warranty claims need the order number and a photo of the defect."),
    "kb://travel/lisbon": ("Lisbon guide", "Lisbon's airport is 7 km from the centre; the metro red line reaches Saldanha in about 20 minutes. Tram 28 is crowded between 10:00 and 17:00. Most museums close on Mondays. Average October high is 22 C."),
    "kb://travel/berlin": ("Berlin guide", "Berlin Brandenburg airport connects to the centre by the FEX express train in about 30 minutes. Museum Island museums are open Tuesday to Sunday. Many shops close on Sundays. Average October high is 14 C."),
    "kb://travel/paris": ("Paris guide", "From Charles de Gaulle, RER B reaches the centre in about 35 minutes. Many museums close on Monday or Tuesday; the Louvre closes on Tuesdays. Average October high is 16 C."),
    "kb://travel/porto": ("Porto guide", "Porto's airport is linked to the centre by metro line E in about 30 minutes. Port cellars in Vila Nova de Gaia offer tours daily. Average October high is 20 C."),
    "kb://travel/rail-passes": ("Rail passes", "A 5-day flexible rail pass within one month costs 283.00 for adults in second class. Seat reservations on high-speed trains cost extra, typically 10.00 to 35.00 per journey."),
    "kb://travel/insurance": ("Travel insurance", "Basic travel cover costs 3.20 per day and includes medical expenses up to 1,000,000.00. Cancellation cover is an add-on at 1.10 per day. Pre-existing conditions must be declared."),
    "kb://travel/hotels": ("Partner hotels", "Partner hotels offer free cancellation up to 48 hours before arrival. Breakfast is included at Casa Azul (Lisbon), Hotel Spree (Berlin) and Maison Lune (Paris). Late checkout until 14:00 is available on request for Plus members."),
    "kb://products/tent-alpine-2": ("Alpine 2 tent", "Two-person 4-season tent, 2.4 kg, 3,000 mm hydrostatic head. Price 389.00. Compatible with the Alpine footprint (sold separately, 45.00)."),
    "kb://products/jacket-storm": ("Storm jacket", "Waterproof shell, 20,000 mm, 410 g. Price 229.00. Sizes XS-XXL. Machine washable at 30 C."),
    "kb://products/boots-ridge": ("Ridge boots", "Leather hiking boots, waterproof membrane, 1.3 kg per pair. Price 179.00. Half sizes available from 38 to 47."),
    "kb://products/stove-micro": ("Micro stove", "Canister stove, 83 g, boils 1 L in 3.5 minutes. Price 64.50. Canisters cannot be shipped by air."),
    "kb://products/pack-trail-40": ("Trail 40 pack", "40-litre backpack, 1.1 kg, adjustable back length. Price 149.00. Rain cover included."),
    "kb://support/contact": ("Contacting support", "Support is available 08:00-20:00 CET on weekdays and 10:00-16:00 on Saturdays. Plus members have a priority queue. Average response time by email is 6 hours."),
    "kb://support/address-change": ("Changing a delivery address", "The delivery address can be changed in the order page until the order is dispatched. After dispatch, contact support to request a carrier redirect, which is not guaranteed."),
    "kb://support/gift-cards": ("Gift cards", "Gift cards are valid for 3 years and can be combined with one other payment method. They cannot be used to buy Plus membership."),
}


def _order_db(seed: int = 7) -> dict[str, dict]:
    rng = random.Random(seed)
    items = ["Alpine 2 tent", "Storm jacket", "Ridge boots", "Micro stove", "Trail 40 pack"]
    prices = {"Alpine 2 tent": 389.00, "Storm jacket": 229.00, "Ridge boots": 179.00, "Micro stove": 64.50, "Trail 40 pack": 149.00}
    statuses = ["processing", "dispatched", "delivered", "returned"]
    orders = {}
    base = date(2026, 8, 1)
    for _ in range(40):
        order_id = "".join(rng.choice(_ALPHABET) for _ in range(12))
        chosen = rng.sample(items, rng.randint(1, 3))
        placed = base + timedelta(days=rng.randint(0, 80))
        orders[order_id] = {
            "order_id": order_id,
            "placed_on": placed.isoformat(),
            "status": rng.choice(statuses),
            "items": [{"name": n, "price": prices[n]} for n in chosen],
            "total": round(sum(prices[n] for n in chosen), 2),
            "ship_to": rng.choice(["home address", "office address", "pickup point"]),
        }
    return orders


ORDERS = _order_db()

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / % ** are allowed")


def builtin_tools(data_dir: Path, flaky: bool = True, seed: int = 0) -> list[ToolSpec]:
    data_dir.mkdir(parents=True, exist_ok=True)
    notes_path = data_dir / "notes.jsonl"
    doc_ids = list(DOCS)
    bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", (DOCS[d][0] + " " + DOCS[d][1]).lower()) for d in doc_ids])
    calls = {"calendar": 0}

    def lookup_order(args: dict) -> ToolResult:
        order_id = str(args["order_id"]).strip().upper()
        order = ORDERS.get(order_id)
        if order is None:
            raise KeyError(f"no order with id {order_id}")
        return ToolResult(ok=True, content=json.dumps(order))

    def search_docs(args: dict) -> ToolResult:
        query = re.findall(r"[a-z0-9]+", str(args["query"]).lower())
        limit = int(args.get("limit", 3))
        scores = bm25.get_scores(query) if query else [0] * len(doc_ids)
        ranked = sorted(zip(scores, doc_ids), key=lambda pair: -pair[0])[:limit]
        hits = [{"id": d, "title": DOCS[d][0], "snippet": DOCS[d][1][:240]} for s, d in ranked if s > 0]
        return ToolResult(ok=True, content=json.dumps(hits))

    def calendar_free_slots(args: dict) -> ToolResult:
        calls["calendar"] += 1
        day = datetime.strptime(str(args["date"]), "%Y-%m-%d").date()
        duration = int(args.get("duration_minutes", 60))
        digest = int(hashlib.sha256(f"{seed}:{day}:{calls['calendar']}".encode()).hexdigest(), 16)
        if flaky and digest % 9 == 0:
            time.sleep(0.2)
            raise TimeoutError("calendar backend did not respond")
        if day.weekday() >= 5:
            return ToolResult(ok=True, content=json.dumps({"date": day.isoformat(), "slots": []}))
        rng = random.Random(f"{seed}:{day}")
        starts = sorted(rng.sample(range(9, 17), 3))
        slots = [f"{h:02d}:00-{(h * 60 + duration) // 60:02d}:{(h * 60 + duration) % 60:02d}" for h in starts]
        return ToolResult(ok=True, content=json.dumps({"date": day.isoformat(), "slots": slots}))

    def calculate(args: dict) -> ToolResult:
        value = _eval(ast.parse(str(args["expression"]), mode="eval"))
        return ToolResult(ok=True, content=str(round(value, 6)))

    def save_note(args: dict) -> ToolResult:
        record = {"title": str(args["title"]), "body": str(args["body"])}
        with open(notes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return ToolResult(ok=True, content=f"saved note '{record['title']}'")

    def fetch_url(args: dict) -> ToolResult:
        url = str(args["url"]).strip()
        if url not in DOCS:
            return ToolResult(ok=False, content="", error=f"404 not found: {url}")
        title, body = DOCS[url]
        return ToolResult(ok=True, content=f"# {title}\n\n{body}")

    return [
        ToolSpec(
            name="lookup_order",
            description="Look up an order by its 12-character order id. Returns status, items, total, and delivery address type.",
            input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
            handler=lookup_order,
        ),
        ToolSpec(
            name="search_docs",
            description="Search the company knowledge base (policies, travel guides, products, support). Returns document ids, titles, and snippets.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 8}},
                "required": ["query"],
            },
            handler=search_docs,
        ),
        ToolSpec(
            name="calendar_free_slots",
            description="List the user's free calendar slots on a date (YYYY-MM-DD) for a meeting of the given length.",
            input_schema={
                "type": "object",
                "properties": {"date": {"type": "string"}, "duration_minutes": {"type": "integer"}},
                "required": ["date"],
            },
            handler=calendar_free_slots,
        ),
        ToolSpec(
            name="calculate",
            description="Evaluate an arithmetic expression with + - * / % ** and parentheses.",
            input_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            handler=calculate,
        ),
        ToolSpec(
            name="save_note",
            description="Save a note for the user with a title and body.",
            input_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                "required": ["title", "body"],
            },
            handler=save_note,
        ),
        ToolSpec(
            name="fetch_url",
            description="Fetch the full text of a knowledge-base document by its id (kb://...).",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=fetch_url,
        ),
    ]
