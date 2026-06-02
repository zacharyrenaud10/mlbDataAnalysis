with open("mlb_analytics/models/player_props_model.py", encoding="utf-8") as f:
    content = f.read()

content = content.replace("max_iter=1000,", "max_iter=3000,")
content = content.replace("max_iter=500,", "max_iter=1000,")

with open("mlb_analytics/models/player_props_model.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed!")