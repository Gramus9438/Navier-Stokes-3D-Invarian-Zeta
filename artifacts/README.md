# Artifacts — Résultats archivés

Ce dossier contient les **résultats complets** des campagnes de simulations (benchmarks).

---

## 📊 Structure

Chaque campagne crée automatiquement un sous-dossier horodaté :

📁 artifacts/ └── 📁 run-YYYYMMDD-HHMMSS/ ├── beltrami.csv ├── taylor_green.csv ├── lamb_oseen.csv ├── ... └── double_opposite.csv

Chaque fichier `.csv` contient les séries temporelles suivantes :  
`t, energy, enstrophy, helicity, H12, zeta`.

---

## 🚀 Comment générer

Lancer tous les cas d’un coup :

```bash
python scripts/bench_all.py --N 32 --steps 80 --dt 0.005 --nu 0.01
