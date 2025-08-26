# Notebooks — Démonstrations et visualisations

Ce dossier contient des **notebooks Jupyter** (`.ipynb`) pour :

- Charger et analyser les résultats stockés dans `data/` ou `artifacts/`.  
- Visualiser l’évolution des grandeurs physiques : énergie, enstrophie, hélicité, norme H¹ᐟ² et invariant ζ(t).  
- Fournir des exemples reproductibles à destination des chercheurs et étudiants.

---

## 🚀 Exemple d’utilisation

1. Lancer Jupyter Lab :

```bash
jupyter lab

2. Ouvrir un notebook (par ex. analysis.ipynb).


3. Importer et tracer un fichier de résultats CSV :



import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/taylor_green_N32_nu0.01.csv")
plt.plot(df["t"], df["zeta"], label="ζ(t)")
plt.xlabel("t")
plt.ylabel("ζ(t)")
plt.legend()
plt.show()

