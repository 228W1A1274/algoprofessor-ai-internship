"""
SkyAssist — Standalone Python version
Run: python skyassist_app.py
Open: http://localhost:7860

Install first:
    pip install openai openai-whisper gradio pydantic python-dotenv tiktoken pillow
    
Create .env file:
    OPENAI_API_KEY=sk-your-key-here
"""

import os, json, base64, traceback
from typing import Optional, Literal
from dataclasses import dataclass
from openai import OpenAI
from pydantic import BaseModel, Field
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

try:
    import whisper as whisper_module
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WHISPER_MODEL = whisper_module.load_model("base")
    WHISPER_OK = True
    print(f"✅ Whisper loaded on {DEVICE.upper()}")
except Exception as e:
    WHISPER_OK = False
    print(f"⚠️ Whisper not available: {e}")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

PRICING = {
    "gpt-4o":      {"input": 2.50/1_000_000, "output": 10.00/1_000_000},
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output":  0.60/1_000_000},
}

@dataclass
class CostTracker:
    session_cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0
    def record(self, model, usage):
        p = PRICING.get(model, {"input":0,"output":0})
        cost = usage.prompt_tokens*p["input"] + usage.completion_tokens*p["output"]
        self.session_cost += cost; self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens; self.call_count += 1
        return cost
    def summary(self):
        return f"💰 ${self.session_cost:.4f} | Tokens: {self.input_tokens+self.output_tokens:,} | Calls: {self.call_count}"

# ── Database ──────────────────────────────────────────────────────────────────
FLIGHT_DB = {
    "SK101": {"origin":"JFK","destination":"LHR","departure":"2025-08-15 09:00","arrival":"2025-08-15 21:00","status":"On Time","seats":{"economy":23,"business":5,"first":2},"price":{"economy":450,"business":1800,"first":4500}},
    "SK202": {"origin":"LHR","destination":"DXB","departure":"2025-08-16 14:30","arrival":"2025-08-16 23:45","status":"Delayed 30 min","seats":{"economy":8,"business":3,"first":1},"price":{"economy":380,"business":1500,"first":3800}},
    "SK303": {"origin":"DXB","destination":"SIN","departure":"2025-08-17 02:15","arrival":"2025-08-17 14:30","status":"On Time","seats":{"economy":45,"business":12,"first":4},"price":{"economy":520,"business":2100,"first":5200}},
}
BOOKING_DB = {
    "SKY123": {"passenger":"James Wilson","flight":"SK101","seat":"24B","class":"economy","status":"Confirmed","checkin_open":True,"baggage":"1×23kg + 7kg carry-on","meal":"Standard"},
    "SKY456": {"passenger":"Priya Sharma","flight":"SK202","seat":"4A","class":"business","status":"Confirmed","checkin_open":False,"baggage":"2×32kg + 15kg carry-on","meal":"Vegetarian"},
    "SKY789": {"passenger":"Marcus Chen","flight":"SK303","seat":"1C","class":"first","status":"Confirmed","checkin_open":True,"baggage":"3×32kg + 20kg carry-on","meal":"Halal"},
}
AVAILABLE_SEATS = {
    "SK101": {"economy":["22A","22C","25F","31C","31D"],"business":["3A","3C","5B"],"first":["1A","2B"]},
    "SK202": {"economy":["18A","19C","20D"],"business":["2A","4C"],"first":["1B"]},
}
BAGGAGE_POLICY = {
    "economy":  {"carry_on":"1 bag max 7kg","checked":"1×23kg included. Extra $60.","oversize":"23-32kg: $100.","sports":"$75."},
    "business": {"carry_on":"2 bags max 15kg","checked":"2×32kg included.","oversize":"Up to 40kg: $150.","sports":"Free."},
    "first":    {"carry_on":"2 bags max 20kg","checked":"3×32kg included.","oversize":"Up to 50kg free.","sports":"Free, priority."},
}

# ── Pydantic Models ───────────────────────────────────────────────────────────
class FlightSearchParams(BaseModel):
    origin: str = Field(..., description="IATA code e.g. JFK")
    destination: str = Field(..., description="IATA code e.g. LHR")
    date: str = Field(..., description="YYYY-MM-DD")
    cabin_class: Literal["economy","business","first"] = Field(default="economy")

class BookingLookupParams(BaseModel):
    booking_ref: str = Field(..., description="Booking reference e.g. SKY123")

class SeatUpgradeParams(BaseModel):
    booking_ref: str = Field(..., description="Booking reference")
    new_seat: str = Field(..., description="Seat number e.g. 3A")
    new_class: Optional[Literal["economy","business","first"]] = Field(None)

class BaggagePolicyParams(BaseModel):
    cabin_class: Literal["economy","business","first"] = Field(...)
    query_type: Literal["carry_on","checked","oversize","sports","all"] = Field(default="all")

def make_tool(cls, name, desc):
    schema = cls.model_json_schema(); schema["additionalProperties"] = False
    return {"type":"function","function":{"name":name,"description":desc,"parameters":schema}}

TOOLS = [
    make_tool(FlightSearchParams,"search_flights","Search flights. Use when asked about schedules or availability."),
    make_tool(BookingLookupParams,"lookup_booking","Look up booking details by reference code."),
    make_tool(SeatUpgradeParams,"upgrade_seat","Change or upgrade a passenger seat."),
    make_tool(BaggagePolicyParams,"get_baggage_policy","Get baggage rules by cabin class."),
]

# ── Tool Functions ────────────────────────────────────────────────────────────
def search_flights(origin,destination,date,cabin_class="economy"):
    o,d = origin.upper(),destination.upper()
    results = [{"flight_number":k,"route":f"{o}→{d}","departure":v["departure"],"arrival":v["arrival"],"status":v["status"],"seats":v["seats"].get(cabin_class,0),"price_usd":v["price"].get(cabin_class,0),"cabin_class":cabin_class} for k,v in FLIGHT_DB.items() if v["origin"]==o and v["destination"]==d and v["seats"].get(cabin_class,0)>0]
    return {"status":"success","flights":results} if results else {"status":"no_flights","message":f"No {cabin_class} flights {o}→{d} on {date}."}

def lookup_booking(booking_ref):
    r = booking_ref.upper()
    if r not in BOOKING_DB: return {"status":"not_found","message":f"Booking {r} not found."}
    b = BOOKING_DB[r]; f = FLIGHT_DB.get(b["flight"],{})
    return {"status":"success","booking_ref":r,"passenger":b["passenger"],"booking_status":b["status"],"flight":b["flight"],"flight_status":f.get("status","?"),"route":f"{f.get('origin','?')}→{f.get('destination','?')}","departure":f.get("departure","N/A"),"seat":b["seat"],"cabin_class":b["class"],"checkin_open":b["checkin_open"],"baggage":b["baggage"],"meal":b["meal"]}

def upgrade_seat(booking_ref,new_seat,new_class=None):
    r = booking_ref.upper()
    if r not in BOOKING_DB: return {"status":"error","message":f"Booking {r} not found."}
    b = BOOKING_DB[r]
    if b["status"]=="Cancelled": return {"status":"error","message":"Cannot modify cancelled booking."}
    tc = new_class or b["class"]
    avail = AVAILABLE_SEATS.get(b["flight"],{}).get(tc,[])
    if new_seat not in avail: return {"status":"seat_unavailable","message":f"Seat {new_seat} not available in {tc}.","available":avail[:6]}
    costs = {"economy→business":800,"economy→first":2000,"business→first":1500}
    cost = costs.get(f"{b['class']}→{tc}",0) if new_class else 0
    old = b["seat"]; BOOKING_DB[r]["seat"]=new_seat
    if new_class: BOOKING_DB[r]["class"]=new_class
    return {"status":"success","old_seat":old,"new_seat":new_seat,"cabin_class":tc,"upgrade_cost_usd":cost,"message":f"Seat changed {old}→{new_seat}."+( f" Fee: ${cost}." if cost else "")}

def get_baggage_policy(cabin_class,query_type="all"):
    if cabin_class not in BAGGAGE_POLICY: return {"status":"error","message":"Unknown class."}
    p = BAGGAGE_POLICY[cabin_class]
    return {"status":"success","cabin_class":cabin_class,"policy":p} if query_type=="all" else {"status":"success","rule":p.get(query_type,"Not found.")}

TOOL_REGISTRY = {"search_flights":search_flights,"lookup_booking":lookup_booking,"upgrade_seat":upgrade_seat,"get_baggage_policy":get_baggage_policy}
VALIDATORS = {"search_flights":FlightSearchParams,"lookup_booking":BookingLookupParams,"upgrade_seat":SeatUpgradeParams,"get_baggage_policy":BaggagePolicyParams}

# ── Memory ────────────────────────────────────────────────────────────────────
class EntityStore:
    def __init__(self): self.entities = {}
    def update(self, d):
        for k,v in d.items():
            if v is not None and v != "" and v != []: self.entities[k]=v
    def to_context(self):
        if not self.entities: return ""
        return "\n[PASSENGER CONTEXT]:\n" + "\n".join(f"  {k}: {v}" for k,v in self.entities.items())
    def clear(self): self.entities = {}

def extract_entities(text, tracker):
    try:
        r = client.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":f'Extract travel entities. Return JSON only, null for missing.\nText: "{text}"\nReturn: {{"passenger_name":null,"booking_ref":null,"flight_number":null,"seat_number":null,"destination":null}}'}],response_format={"type":"json_object"},temperature=0,max_tokens=120)
        tracker.record("gpt-4o-mini",r.usage); return json.loads(r.choices[0].message.content)
    except: return {}

# ── Multimodal ────────────────────────────────────────────────────────────────
def transcribe_audio(path):
    if not WHISPER_OK or path is None: return ""
    try: return WHISPER_MODEL.transcribe(path,fp16=(DEVICE=="cuda"),language="en")["text"].strip()
    except: return ""

def analyze_image(path, tracker):
    if path is None: return ""
    try:
        with open(path,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        ext = path.split(".")[-1].lower(); mt = {"jpg":"jpeg","jpeg":"jpeg","png":"png"}.get(ext,"jpeg")
        r = client.chat.completions.create(model="gpt-4o",messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/{mt};base64,{b64}","detail":"high"}},{"type":"text","text":"Extract all boarding pass information: passenger name, flight number, route, departure time, gate, seat, booking reference."}]}],max_tokens=400)
        tracker.record("gpt-4o",r.usage); return r.choices[0].message.content
    except: return ""

# ── Agent ─────────────────────────────────────────────────────────────────────
SYSTEM = """You are SkyAssist, a professional airline support AI.
Be helpful, warm, and clear. Use tools to get real data before answering.
Never guess booking or flight information — always call a tool.
{ctx}"""

def run_agent(msg, history, entity_store, tracker):
    history.append({"role":"user","content":msg})
    trimmed = history[-10:]
    messages = [{"role":"system","content":SYSTEM.format(ctx=entity_store.to_context())}] + trimmed
    for _ in range(6):
        r = client.chat.completions.create(model="gpt-4o",messages=messages,tools=TOOLS,tool_choice="auto",temperature=0.3,max_tokens=800)
        tracker.record("gpt-4o",r.usage)
        m = r.choices[0].message
        md = {"role":"assistant","content":m.content or ""}
        if m.tool_calls: md["tool_calls"]=[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in m.tool_calls]
        messages.append(md); history.append(md)
        if not m.tool_calls: return m.content or "Please try again.", history
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                if tc.function.name in VALIDATORS: args = VALIDATORS[tc.function.name](**args).model_dump(exclude_none=True)
                result = TOOL_REGISTRY[tc.function.name](**args)
            except Exception as e: result = {"error":str(e)}
            tm = {"role":"tool","tool_call_id":tc.id,"content":json.dumps(result)}
            messages.append(tm); history.append(tm)
    return "Max steps reached. Please rephrase.", history

# ── Gradio App ────────────────────────────────────────────────────────────────
def build_app(api_key=""):
    if api_key: client.api_key = api_key
    history, entity_store, tracker = [], EntityStore(), CostTracker()

    def process(text, audio, image, chat_history):
        nonlocal history, entity_store, tracker
        parts = []
        if text and text.strip(): parts.append(text.strip())
        if audio:
            t = transcribe_audio(audio)
            if t: parts.append(f"[Voice]: {t}")
        if image:
            s = analyze_image(image, tracker)
            if s: parts.append(f"[Boarding Pass]:\n{s}\nPlease help with this boarding pass.")
        if not parts:
            chat_history.append(("(empty)", "Please type, record, or upload a boarding pass."))
            return chat_history, "", tracker.summary()
        user_msg = "\n\n".join(parts)
        entity_store.update(extract_entities(user_msg, tracker))
        try: response, history = run_agent(user_msg, history, entity_store, tracker)
        except Exception as e: response = f"Error: {e}"; traceback.print_exc()
        label = text.strip() if text else ""
        if audio: label += (" " if label else "") + "🎙️"
        if image: label += (" " if label else "") + "📋"
        chat_history.append((label, response))
        return chat_history, "", tracker.summary()

    def reset():
        nonlocal history, entity_store, tracker
        history, entity_store, tracker = [], EntityStore(), CostTracker()
        return [], "Session reset."

    with gr.Blocks(title="✈️ SkyAssist", theme=gr.themes.Base(primary_hue="blue")) as demo:
        gr.HTML('<div style="background:linear-gradient(135deg,#0D47A1,#0277BD);padding:20px;border-radius:12px;text-align:center;color:white;margin-bottom:12px"><h1 style="margin:0;letter-spacing:3px">✈️ SKYASSIST</h1><p style="margin:6px 0 0;opacity:.85">Multi-Modal Airline Support Agent | GPT-4o · Whisper · Vision</p></div>')
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="💬 Conversation", height=450, bubble_full_width=False)
                with gr.Row():
                    text_input = gr.Textbox(label="Your message", placeholder="Ask about flights, bookings, seats, or baggage...", lines=2, scale=5)
                    send_btn = gr.Button("Send ✈️", variant="primary", scale=1)
                cost_box = gr.Textbox(label="📊 Usage", value="No calls yet", interactive=False)
            with gr.Column(scale=2):
                gr.Markdown("### 🎙️ Voice Input"); audio_input = gr.Audio(sources=["microphone"], type="filepath")
                gr.Markdown("### 📋 Boarding Pass"); image_input = gr.Image(type="filepath", height=160)
                gr.Examples(examples=[["Look up my booking SKY123",None,None],["Business flights JFK to LHR August 15?",None,None],["Upgrade SKY123 seat to 3A business class",None,None],["Economy baggage allowance?",None,None]],inputs=[text_input,audio_input,image_input])
                reset_btn = gr.Button("🔄 Reset", variant="stop")
                reset_status = gr.Textbox(label="", interactive=False, max_lines=1)
        for trigger in [send_btn.click, text_input.submit]:
            trigger(fn=process, inputs=[text_input,audio_input,image_input,chatbot], outputs=[chatbot,text_input,cost_box])
        reset_btn.click(fn=reset, outputs=[chatbot, reset_status])
    return demo

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY in .env file")
    app = build_app()
    app.launch(server_port=7860, share=False, show_error=True)
