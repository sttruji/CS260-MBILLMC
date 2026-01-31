# Setup 

**Environment** 
python -m venv .venv 
pip install -r requirements.txt 

cp env.example .env

Github > Settings > Developer Settings > Personal access token > Tokens (Classic)
Generate new token > Scope (public_repo / read:user)

Place key in .env 


**Scripts** 

python scripts/collect_github.py 
    