import json

with open("cache/prediction_results.json") as f:
    data = json.load(f)

all_graded = []
for day in data["games"]:
    all_graded.extend(day.get("graded", []))

# Remove 0-0 games (not finished)
all_graded = [g for g in all_graded if g.get("score", "0-0") != "0-0"]

tiers = {
    "70%+":   [g for g in all_graded if g["model_prob"] >= 0.70],
    "65-70%": [g for g in all_graded if 0.65 <= g["model_prob"] < 0.70],
    "60-65%": [g for g in all_graded if 0.60 <= g["model_prob"] < 0.65],
    "55-60%": [g for g in all_graded if 0.55 <= g["model_prob"] < 0.60],
    "<55%":   [g for g in all_graded if g["model_prob"] < 0.55],
}

def pct(lst):
    if not lst: return "N/A"
    c = sum(1 for g in lst if g["correct"])
    return f"{c}/{len(lst)} ({c/len(lst)*100:.0f}%)"

print(f"\nOverall: {pct(all_graded)}")
print(f"\nBy tier:")
for tier, games in tiers.items():
    print(f"  {tier:<10} {pct(games)}")

# Home vs away
home = [g for g in all_graded 
        if g["matchup"].split(" @ ")[1].strip() == g["model_pick"]]
away = [g for g in all_graded
        if g["matchup"].split(" @ ")[0].strip() == g["model_pick"]]
print(f"\nHome picks: {pct(home)}")
print(f"Away picks: {pct(away)}")