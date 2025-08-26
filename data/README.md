# Data — Résultats simples

Ce dossier contient les résultats **légers** (fichiers CSV) générés par les simulations individuelles exécutées avec `scripts/run_case.py`.

---

## 📊 Structure

Chaque exécution crée un fichier `.csv` nommé selon le cas et les paramètres. Exemple :

📁 data/ ├── taylor_green_N32_nu0.01.csv └── beltrami_N48_nu0.005.csv

Chaque fichier `.csv` contient les colonnes :  
`t, energy, enstrophy, helicity, H12, zeta`

---

## 🚀 Comment générer

Exemple d’exécution :

```bash
python scripts/run_case.py --case taylor_green --N 32 --steps 50 --dt 0.01 --nu 0.01

Résultat → data/taylor_green_N32_nu0.01.csv.

