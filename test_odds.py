from dotenv import load_dotenv
load_dotenv()
from mlb_analytics.ingestion.fanduel_scraper import fetch_player_props_api, fetch_moneylines_api

print("=== MONEYLINES ===")
ml = fetch_moneylines_api()
print(ml)

print("=== PROPS ===")
props = fetch_player_props_api()
if props.empty:
    print("NO PROPS FOUND")
else:
    print(props.head(20))