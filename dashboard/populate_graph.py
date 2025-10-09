import json, random

# Use this script to make a mock graph to populate index.html
# graph_mock.json --- After using to_graphjson.py
with open("dashboard/graph_mock.json") as f:
    g = json.load(f)

# === Communities and Risk Taxonomy ===
communities = ["Trading", "Legal", "Executive", "Finance", "PR"]
risk_categories = [
    "Compliance Risk",
    "Operational Risk",
    "Strategic Risk",
    "Fraud / Ethical Risk",
    "Reputational Risk"
]

for n in g["nodes"]:
    n["community"] = random.choice(communities)
    risk = round(random.uniform(0, 1), 2)
    sentiment = round(random.uniform(-1, 1), 2)
    n["avg_risk"] = risk
    n["avg_sentiment"] = sentiment
    
    if risk > 0.75:
        n["risk_category"] = random.choice(["Fraud / Ethical Risk", "Compliance Risk"])
    elif risk > 0.5:
        n["risk_category"] = random.choice(["Operational Risk", "Strategic Risk"])
    else:
        n["risk_category"] = random.choice(["Reputational Risk", "Operational Risk"])

for e in g["edges"]:
    e["risk_score"] = round(random.uniform(0, 1), 2)
    e["weight"] = round(random.uniform(0.2, 1.5), 2)
    e["sentiment_score"] = round(random.uniform(-1, 1), 2)

with open("dashboard/graph_mock.json", "w") as f:
    json.dump(g, f, indent=2)

print(f"Enhanced graph saved with {len(g['nodes'])} nodes and {len(g['edges'])} edges.")

