import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model (must sit next to app.py)
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_medical_cost_predict.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact feature order the model was trained on:
FEATURES = ["age", "bmi", "children", "smoker", "region"]

# Label encodings (alphabetical, as produced by sklearn LabelEncoder / pandas codes).
# If you trained with a different mapping, change it here.
SMOKER_MAP = {"No": 0, "Yes": 1}
REGION_MAP = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

DEFAULTS = {"age": 35, "bmi": 26.5, "children": 1, "smoker": "No", "region": "Northeast"}


def bmi_category(bmi: float):
    if bmi < 18.5:
        return "Underweight", "warn"
    if bmi < 25:
        return "Healthy weight", "good"
    if bmi < 30:
        return "Overweight", "warn"
    return "Obese range", "bad"


def parse_inputs(form):
    """Validate form values and return a clean dict."""
    age = int(form.get("age", DEFAULTS["age"]))
    bmi = float(form.get("bmi", DEFAULTS["bmi"]))
    children = int(form.get("children", DEFAULTS["children"]))
    smoker = form.get("smoker", DEFAULTS["smoker"])
    region = form.get("region", DEFAULTS["region"])

    if not (18 <= age <= 100):
        raise ValueError("Age must be between 18 and 100.")
    if not (10 <= bmi <= 60):
        raise ValueError("BMI must be between 10 and 60.")
    if not (0 <= children <= 10):
        raise ValueError("Children must be between 0 and 10.")
    if smoker not in SMOKER_MAP or region not in REGION_MAP:
        raise ValueError("Invalid smoker or region selection.")
    return {"age": age, "bmi": bmi, "children": children, "smoker": smoker, "region": region}


PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MediCost AI · Medical Cost Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f2f8f6;
    --surface: #ffffff;
    --ink: #0f2a2a;
    --muted: #5d7777;
    --line: #dbe9e5;
    --teal-50: #e6f6f2;
    --teal-100: #c8ece3;
    --teal-500: #14b8a6;
    --teal-600: #0d9488;
    --teal-700: #0f766e;
    --teal-900: #0b3d3a;
    --good: #16a34a; --warn: #d97706; --bad: #dc2626;
    --radius: 20px;
    --shadow: 0 10px 40px -12px rgba(13, 148, 136, .25);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0a1816; --surface: #112624; --ink: #e8f5f2; --muted: #8fb0aa;
      --line: #1d3a36; --teal-50: #12312d; --teal-100: #17433d;
      --shadow: 0 10px 40px -12px rgba(0,0,0,.6);
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    background:
      radial-gradient(900px 500px at 100% -10%, rgba(20,184,166,.18), transparent 60%),
      radial-gradient(700px 400px at -10% 110%, rgba(45,212,191,.14), transparent 60%),
      var(--bg);
    color: var(--ink);
    min-height: 100vh;
  }
  .layout { display: grid; grid-template-columns: 360px 1fr; min-height: 100vh; }

  /* ---------- Sidebar ---------- */
  .sidebar {
    background: var(--surface);
    border-right: 1px solid var(--line);
    padding: 28px 26px;
    position: sticky; top: 0; height: 100vh; overflow-y: auto;
  }
  .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
  .logo {
    width: 42px; height: 42px; border-radius: 13px;
    background: linear-gradient(135deg, var(--teal-500), var(--teal-700));
    display: grid; place-items: center; color: #fff;
    box-shadow: 0 8px 20px -6px rgba(13,148,136,.6);
  }
  .brand h1 { font-size: 1.2rem; font-weight: 800; letter-spacing: -.02em; }
  .brand small { color: var(--muted); font-size: .78rem; display: block; font-weight: 500; }
  .sidebar > p.lead { color: var(--muted); font-size: .88rem; line-height: 1.5; margin: 14px 0 22px; }

  .field { margin-bottom: 22px; }
  .field label.title {
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: var(--muted); margin-bottom: 10px;
  }
  .field .val { color: var(--teal-600); font-size: 1.05rem; font-weight: 800; letter-spacing: 0; text-transform: none; }

  input[type=range] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 8px; border-radius: 99px;
    background: linear-gradient(to right, var(--teal-500) var(--fill, 50%), var(--teal-100) var(--fill, 50%));
    outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 24px; height: 24px; border-radius: 50%;
    background: #fff; border: 4px solid var(--teal-600);
    box-shadow: 0 4px 12px rgba(13,148,136,.45);
  }
  input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%; background: #fff; border: 4px solid var(--teal-600);
  }
  .ticks { display: flex; justify-content: space-between; font-size: .72rem; color: var(--muted); margin-top: 6px; }

  .segment { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; background: var(--teal-50); padding: 5px; border-radius: 14px; }
  .segment input { display: none; }
  .segment label {
    text-align: center; padding: 11px 0; border-radius: 10px; cursor: pointer;
    font-weight: 600; font-size: .92rem; color: var(--muted); transition: all .2s;
  }
  .segment input:checked + label { background: var(--surface); color: var(--teal-700); box-shadow: 0 3px 10px rgba(13,148,136,.2); }
  .segment input[value=Yes]:checked + label { color: var(--bad); }

  select {
    width: 100%; padding: 13px 14px; border-radius: 14px; border: 1.5px solid var(--line);
    background: var(--surface); color: var(--ink); font: inherit; font-weight: 500; cursor: pointer;
  }
  select:focus { outline: none; border-color: var(--teal-500); box-shadow: 0 0 0 4px rgba(20,184,166,.18); }

  .btn {
    width: 100%; border: 0; cursor: pointer; padding: 16px; border-radius: 16px;
    font: inherit; font-weight: 700; font-size: 1rem; color: #fff; margin-top: 6px;
    background: linear-gradient(135deg, var(--teal-500), var(--teal-700));
    box-shadow: 0 12px 28px -8px rgba(13,148,136,.65);
    transition: transform .15s, box-shadow .15s;
  }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 16px 32px -8px rgba(13,148,136,.75); }
  .btn:active { transform: translateY(0); }

  /* ---------- Main ---------- */
  main { padding: 48px clamp(20px, 5vw, 64px); max-width: 980px; }
  .eyebrow {
    display: inline-flex; align-items: center; gap: 8px; background: var(--teal-50); color: var(--teal-700);
    padding: 7px 14px; border-radius: 99px; font-size: .78rem; font-weight: 700; letter-spacing: .04em;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--teal-500); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(20,184,166,.6);} 50% { box-shadow: 0 0 0 7px rgba(20,184,166,0);} }
  h2.hero { font-family: 'Fraunces', Georgia, serif; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.1; margin: 18px 0 12px; letter-spacing: -.02em; }
  h2.hero span { color: var(--teal-600); }
  .sub { color: var(--muted); max-width: 560px; line-height: 1.6; margin-bottom: 34px; }

  /* Premium metric card */
  .metric {
    position: relative; overflow: hidden; border-radius: 28px; padding: 40px;
    color: #fff; box-shadow: var(--shadow);
    background:
      radial-gradient(500px 260px at 90% 0%, rgba(255,255,255,.22), transparent 60%),
      linear-gradient(135deg, #14b8a6 0%, #0d9488 45%, #0b3d3a 100%);
    animation: rise .6s ease both;
  }
  .metric::after {
    content: ""; position: absolute; right: -60px; bottom: -80px; width: 260px; height: 260px;
    border-radius: 50%; border: 36px solid rgba(255,255,255,.08);
  }
  @keyframes rise { from { opacity: 0; transform: translateY(16px);} to { opacity: 1; transform: none; } }
  .metric .label { font-size: .8rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; opacity: .85; }
  .metric .amount {
    font-family: 'Fraunces', Georgia, serif; font-weight: 700;
    font-size: clamp(3rem, 8vw, 5.4rem); line-height: 1; margin: 14px 0 6px; letter-spacing: -.03em;
  }
  .metric .amount sup { font-size: .45em; vertical-align: top; position: relative; top: .2em; margin-right: 4px; opacity: .85; }
  .metric .per { opacity: .85; font-size: .95rem; }
  .chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 26px; position: relative; z-index: 1; }
  .chip { background: rgba(255,255,255,.16); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,.25); padding: 8px 14px; border-radius: 99px; font-size: .84rem; font-weight: 600; }

  .empty {
    border: 2px dashed var(--teal-100); border-radius: 28px; padding: 56px 32px; text-align: center; color: var(--muted);
    background: var(--surface);
  }
  .empty svg { color: var(--teal-500); margin-bottom: 14px; }
  .empty h3 { color: var(--ink); margin-bottom: 6px; }

  .error { background: #fee2e2; color: #991b1b; padding: 14px 18px; border-radius: 14px; margin-bottom: 20px; font-weight: 500; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 22px; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px 22px; }
  .card .k { font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
  .card .v { font-size: 1.5rem; font-weight: 800; margin-top: 8px; letter-spacing: -.02em; }
  .card .s { font-size: .85rem; margin-top: 4px; font-weight: 600; }
  .good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); }

  .vector { margin-top: 22px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 22px; }
  .vector .k { font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
  .vector code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--teal-700); font-size: .92rem; word-break: break-all; }
  .vector .cols { color: var(--muted); font-size: .8rem; margin-top: 6px; }

  .disclaimer { margin-top: 30px; font-size: .8rem; color: var(--muted); line-height: 1.6; }

  @media (max-width: 860px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
    main { padding-top: 32px; }
    .metric { padding: 30px 24px; }
  }
</style>
</head>
<body>
<div class="layout">

  <!-- ================= SIDEBAR ================= -->
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l2-6 4 12 2-6h6"/></svg>
      </div>
      <div><h1>MediCost AI</h1><small>Health cost estimator</small></div>
    </div>
    <p class="lead">Adjust your profile below, then estimate your yearly medical insurance charges.</p>

    <form method="POST" action="/" id="form">
      <div class="field">
        <label class="title" for="age">Age <span class="val" id="age_v">{{ v.age }}</span></label>
        <input type="range" id="age" name="age" min="18" max="100" step="1" value="{{ v.age }}">
        <div class="ticks"><span>18</span><span>100</span></div>
      </div>

      <div class="field">
        <label class="title" for="bmi">BMI <span class="val" id="bmi_v">{{ '%.1f' % v.bmi }}</span></label>
        <input type="range" id="bmi" name="bmi" min="10" max="60" step="0.1" value="{{ v.bmi }}">
        <div class="ticks"><span>10</span><span>60</span></div>
      </div>

      <div class="field">
        <label class="title" for="children">Children <span class="val" id="children_v">{{ v.children }}</span></label>
        <input type="range" id="children" name="children" min="0" max="10" step="1" value="{{ v.children }}">
        <div class="ticks"><span>0</span><span>10</span></div>
      </div>

      <div class="field">
        <label class="title">Smoker</label>
        <div class="segment">
          <input type="radio" name="smoker" id="s_no" value="No" {{ 'checked' if v.smoker == 'No' }}>
          <label for="s_no">🚭 No</label>
          <input type="radio" name="smoker" id="s_yes" value="Yes" {{ 'checked' if v.smoker == 'Yes' }}>
          <label for="s_yes">🚬 Yes</label>
        </div>
      </div>

      <div class="field">
        <label class="title" for="region">Region</label>
        <select id="region" name="region">
          {% for r in regions %}
          <option value="{{ r }}" {{ 'selected' if v.region == r }}>{{ r }}</option>
          {% endfor %}
        </select>
      </div>

      <button class="btn" type="submit">Predict my cost →</button>
    </form>
  </aside>

  <!-- ================= MAIN ================= -->
  <main>
    <span class="eyebrow"><span class="dot"></span> Linear regression model</span>
    <h2 class="hero">Know your <span>medical costs</span> before they surprise you.</h2>
    <p class="sub">A simple, transparent model estimates annual insurance charges from age, BMI, family size, smoking status and region.</p>

    {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}

    {% if result %}
      <section class="metric">
        <div class="label">Estimated annual medical cost</div>
        <div class="amount"><sup>$</sup>{{ '{:,.2f}'.format(result.cost) }}</div>
        <div class="per">≈ ${{ '{:,.0f}'.format(result.cost / 12) }} per month</div>
        <div class="chips">
          <span class="chip">🎂 {{ v.age }} yrs</span>
          <span class="chip">⚖️ BMI {{ '%.1f' % v.bmi }}</span>
          <span class="chip">👨‍👩‍👧 {{ v.children }} child{{ '' if v.children == 1 else 'ren' }}</span>
          <span class="chip">{{ '🚬 Smoker' if v.smoker == 'Yes' else '🚭 Non-smoker' }}</span>
          <span class="chip">📍 {{ v.region }}</span>
        </div>
      </section>

      <div class="grid">
        <div class="card">
          <div class="k">BMI status</div>
          <div class="v">{{ '%.1f' % v.bmi }}</div>
          <div class="s {{ result.bmi_cls }}">{{ result.bmi_label }}</div>
        </div>
        <div class="card">
          <div class="k">Smoking impact</div>
          <div class="v">{{ 'High' if v.smoker == 'Yes' else 'None' }}</div>
          <div class="s {{ 'bad' if v.smoker == 'Yes' else 'good' }}">
            {{ 'Largest cost driver' if v.smoker == 'Yes' else 'Keeps costs lower' }}
          </div>
        </div>
        <div class="card">
          <div class="k">Weekly average</div>
          <div class="v">${{ '{:,.0f}'.format(result.cost / 52) }}</div>
          <div class="s" style="color: var(--muted)">per week</div>
        </div>
      </div>

      <div class="vector">
        <div class="k">Model input (encoded array)</div>
        <code>{{ result.vector }}</code>
        <div class="cols">Columns: age, bmi, children, smoker, region</div>
      </div>
    {% elif not error %}
      <div class="empty">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1 1.1L12 21l7.8-7.5 1-1.1a5.5 5.5 0 0 0 0-7.8z"/></svg>
        <h3>Your estimate will appear here</h3>
        <p>Set your details in the sidebar and press <b>Predict my cost</b>.</p>
      </div>
    {% endif %}

    <p class="disclaimer">
      ⚕️ This is an educational estimate from a statistical model (values in USD) and not medical, financial or insurance advice.
    </p>
  </main>
</div>

<script>
  // Live slider values + filled track
  function bind(id, fmt) {
    const el = document.getElementById(id), out = document.getElementById(id + '_v');
    const update = () => {
      out.textContent = fmt(parseFloat(el.value));
      el.style.setProperty('--fill', ((el.value - el.min) / (el.max - el.min) * 100) + '%');
    };
    el.addEventListener('input', update); update();
  }
  bind('age', v => v.toFixed(0));
  bind('bmi', v => v.toFixed(1));
  bind('children', v => v.toFixed(0));
</script>
</body>
</html>
"""


class Obj(dict):
    __getattr__ = dict.get


@app.route("/", methods=["GET", "POST"])
def index():
    values = dict(DEFAULTS)
    result, error = None, None

    if request.method == "POST":
        try:
            values = parse_inputs(request.form)

            # Build input in the EXACT order/encoding the model was trained on
            row = [
                values["age"],
                values["bmi"],
                values["children"],
                SMOKER_MAP[values["smoker"]],
                REGION_MAP[values["region"]],
            ]
            X = pd.DataFrame([row], columns=FEATURES)  # keeps feature names -> no sklearn warning
            cost = max(0.0, float(model.predict(X)[0]))  # linear models can dip below 0 at extremes

            label, cls = bmi_category(values["bmi"])
            result = Obj(cost=cost, bmi_label=label, bmi_cls=cls, vector=str([row]))
        except (ValueError, TypeError) as e:
            error = str(e) or "Please check your inputs."

    return render_template_string(PAGE, v=Obj(values), regions=list(REGION_MAP), result=result, error=error)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
