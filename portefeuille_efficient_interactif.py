# portfolios_cli.py
# -*- coding: utf-8 -*-
# Script interactif : choix de 2 actifs (USA + Europe), stats & graphes, frontière efficiente + CML
# Dépendances : pandas, numpy, yfinance, matplotlib, scipy

import sys
import math
import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Tentative d'import de SciPy (pour l'optimisation du Sharpe) ---
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

TRADING_DAYS = 252

# ============================
# 1) Univers d'actions (50 env.)
# ============================
UNIVERSE = [
    # --- USA ---
    ("AAPL", "Apple Inc. (US)"),
    ("MSFT", "Microsoft Corp. (US)"),
    ("AMZN", "Amazon.com Inc. (US)"),
    ("GOOGL", "Alphabet Inc. Class A (US)"),
    ("META", "Meta Platforms Inc. (US)"),
    ("NVDA", "NVIDIA Corp. (US)"),
    ("TSLA", "Tesla Inc. (US)"),
    ("JPM", "JPMorgan Chase & Co. (US)"),
    ("JNJ", "Johnson & Johnson (US)"),
    ("XOM", "Exxon Mobil Corp. (US)"),
    ("PG", "Procter & Gamble Co. (US)"),
    ("V", "Visa Inc. (US)"),
    ("MA", "Mastercard Inc. (US)"),
    ("DIS", "Walt Disney Co. (US)"),
    ("NFLX", "Netflix Inc. (US)"),
    ("AMD", "Advanced Micro Devices (US)"),
    ("INTC", "Intel Corp. (US)"),
    ("CSCO", "Cisco Systems (US)"),
    ("KO", "Coca-Cola Co. (US)"),
    ("MCD", "McDonald's Corp. (US)"),
    ("WMT", "Walmart Inc. (US)"),
    ("BAC", "Bank of America (US)"),
    ("PFE", "Pfizer Inc. (US)"),
    ("NKE", "Nike Inc. (US)"),
    ("ABNB", "Airbnb Inc. (US)"),
    # --- France (.PA) ---
    ("MC.PA", "LVMH (FR)"),
    ("OR.PA", "L'Oréal (FR)"),
    ("AI.PA", "Air Liquide (FR)"),
    ("AIR.PA", "Airbus (FR)"),
    ("SAN.PA", "Sanofi (FR)"),
    ("BN.PA", "Danone (FR)"),
    ("SU.PA", "Schneider Electric (FR)"),
    ("DG.PA", "Vinci (FR)"),
    ("RI.PA", "Pernod Ricard (FR)"),
    ("CAP.PA", "Capgemini (FR)"),
    ("BNP.PA", "BNP Paribas (FR)"),
    ("ACA.PA", "Crédit Agricole (FR)"),
    # --- Allemagne (.DE) ---
    ("SAP.DE", "SAP (DE)"),
    ("ALV.DE", "Allianz (DE)"),
    ("SIE.DE", "Siemens (DE)"),
    ("BMW.DE", "BMW (DE)"),
    ("BAS.DE", "BASF (DE)"),
    ("BAYN.DE", "Bayer (DE)"),
    ("DTE.DE", "Deutsche Telekom (DE)"),
    ("ADS.DE", "Adidas (DE)"),
    # --- Royaume-Uni (.L) ---
    ("SHEL.L", "Shell plc (UK)"),
    ("BP.L", "BP plc (UK)"),
    ("HSBA.L", "HSBC Holdings (UK)"),
    ("AZN.L", "AstraZeneca (UK)"),
    ("ULVR.L", "Unilever (UK)"),
    ("DGE.L", "Diageo (UK)"),
    ("RIO.L", "Rio Tinto (UK)"),
    ("LSEG.L", "London Stock Exchange Group (UK)"),
    # --- Suisse (.SW) ---
    ("NESN.SW", "Nestlé (CH)"),
    ("NOVN.SW", "Novartis (CH)"),
    ("ROG.SW", "Roche (CH)"),
    ("UBSG.SW", "UBS Group (CH)"),
    # --- Espagne (.MC) ---
    ("ITX.MC", "Inditex (ES)"),
    ("SAN.MC", "Banco Santander (ES)"),
    ("BBVA.MC", "BBVA (ES)"),
    ("IBE.MC", "Iberdrola (ES)"),
    # --- Italie (.MI) ---
    ("ENEL.MI", "Enel (IT)"),
    ("ENI.MI", "ENI (IT)"),
    ("ISP.MI", "Intesa Sanpaolo (IT)"),
    ("UCG.MI", "UniCredit (IT)"),
    # --- Pays-Bas / Nordics (qq ex.) ---
    ("ASML", "ASML Holding NV (US ADR/Nasdaq)"),
    ("NOKIA.HE", "Nokia (FI)"),
]

# =================================================
# 2) Utilitaires : affichage, saisie, téléchargement
# =================================================
def print_header(title: str):
    bar = "═" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")

def list_universe():
    print_header("Sélectionnez 2 actifs par INDEX")
    # Formatage en colonnes
    width_idx = len(str(len(UNIVERSE) - 1))
    for i, (ticker, name) in enumerate(UNIVERSE):
        print(f"{str(i).rjust(width_idx)} : {ticker.ljust(10)} — {name}")
    print("\nExemple d'entrée : 0, 27   ou   5 12")

def ask_two_indices():
    while True:
        raw = input("\nEntrez deux indices séparés par virgule ou espace : ").strip()
        tokens = [t for t in raw.replace(",", " ").split() if t]
        try:
            idx = list(map(int, tokens))
        except ValueError:
            print("⚠️  Indices invalides. Recommencez.")
            continue
        if len(idx) != 2:
            print("⚠️  Veuillez fournir exactement deux indices.")
            continue
        if any((i < 0 or i >= len(UNIVERSE)) for i in idx):
            print("⚠️  Un indice est hors limites.")
            continue
        if idx[0] == idx[1]:
            print("⚠️  Les deux indices doivent être différents.")
            continue
        return idx[0], idx[1]

def ask_risk_free():
    while True:
        raw = input("Entrez le taux sans risque ANNUEL (ex: 0.02 pour 2%) : ").strip()
        try:
            rf = float(raw)
            return rf
        except ValueError:
            print("⚠️  Valeur non numérique. Réessayez.")

def ask_dates():
    print("\nPériode d'historique (appuyez Entrée pour conserver les défauts)")
    start = input("Date de début [défaut: 2015-01-01] (YYYY-MM-DD) : ").strip() or "2015-01-01"
    end = input("Date de fin   [défaut: aujourd'hui] (YYYY-MM-DD ou vide) : ").strip() or None
    return start, end

def fetch_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"].copy()
    else:
        prices = raw.rename("Adj Close")
    prices = (
        prices.sort_index()
              .dropna(how="all")
              .ffill()  # alignement sur jours fériés
    )
    # S'assurer d'avoir les 2 colonnes
    present = [c for c in prices.columns]
    if len(present) != 2:
        raise RuntimeError(f"Attendu 2 colonnes de prix, obtenu {len(present)} : {present}")
    prices.columns = [str(c) for c in prices.columns]
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return log_ret

# ==================================
# 3) Mesures, tracés et optimisations
# ==================================
def describe_returns(log_returns: pd.DataFrame):
    print_header("Statistiques — rendements log (jours)")
    mean_d = log_returns.mean()
    var_d = log_returns.var()
    corr = log_returns.corr()

    # Annualisation
    mean_a = mean_d * TRADING_DAYS
    cov_a = log_returns.cov() * TRADING_DAYS
    vol_a = np.sqrt(np.diag(cov_a))
    stats = pd.DataFrame({
        "Moyenne_jour": mean_d,
        "Moyenne_ann": mean_a,
        "Vol_ann": vol_a
    })
    print(stats.to_string(float_format=lambda x: f"{x:0.6f}"))
    print("\nCorrélation (journalier) :")
    print(corr.to_string(float_format=lambda x: f"{x:0.4f}"))
    return stats, cov_a, mean_a

def plot_log_returns(log_returns: pd.DataFrame):
    log_returns.plot(title="Rendements log quotidiens")
    plt.xlabel("Date"); plt.ylabel("Rendement log"); plt.grid(True); plt.tight_layout(); plt.show()

def rolling_vol_corr(log_returns: pd.DataFrame, window=30):
    print_header(f"Volatilité & corrélation glissantes (fenêtre {window} jours)")
    vol_30d = log_returns.rolling(window=window).std() * math.sqrt(TRADING_DAYS)
    vol_30d.plot(title=f"Volatilité glissante {window} jours (annualisée)")
    plt.xlabel("Date"); plt.ylabel("Vol annualisée"); plt.grid(True); plt.tight_layout(); plt.show()

    s1, s2 = log_returns.columns[:2]
    corr_30d = log_returns[s1].rolling(window=window).corr(log_returns[s2])
    corr_30d.plot(title=f"Corrélation glissante {window} jours")
    plt.xlabel("Date"); plt.ylabel("Corrélation"); plt.grid(True); plt.tight_layout(); plt.show()

def sample_efficient_frontier(mean_ann: pd.Series, cov_ann: pd.DataFrame, rf_ann: float, n=3000):
    """
    Échantillonne aléatoirement des portefeuilles (2 actifs ici) pour approximer la frontière.
    Retourne: volatilities, returns, sharpes, weights_list
    """
    dim = len(mean_ann)
    vols, rets, sharps, weights_list = [], [], [], []
    for _ in range(n):
        w = np.random.random(dim)
        w /= w.sum()
        port_ret = float(np.dot(mean_ann.values, w))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_ann.values, w))))
        sharpe = (port_ret - rf_ann) / port_vol if port_vol > 0 else np.nan
        vols.append(port_vol); rets.append(port_ret); sharps.append(sharpe); weights_list.append(w)
    return np.array(vols), np.array(rets), np.array(sharps), weights_list

def optimal_sharpe(mean_ann: pd.Series, cov_ann: pd.DataFrame, rf_ann: float):
    """
    Portefeuille max Sharpe via optimisation (si SciPy dispo); sinon approximation par échantillonnage.
    """
    if SCIPY_AVAILABLE:
        dim = len(mean_ann)
        x0 = np.ones(dim) / dim
        bounds = tuple((0.0, 1.0) for _ in range(dim))
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

        def neg_sharpe(w):
            ret = float(np.dot(mean_ann.values, w))
            vol = float(np.sqrt(np.dot(w.T, np.dot(cov_ann.values, w))))
            return - (ret - rf_ann) / vol

        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
        w = res.x
        ret = float(np.dot(mean_ann.values, w))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_ann.values, w))))
        sh = (ret - rf_ann) / vol
        return w, ret, vol, sh, True
    else:
        # Fallback : grid fin + sampling
        vols, rets, sharps, wlist = sample_efficient_frontier(mean_ann, cov_ann, rf_ann, n=20000)
        idx = int(np.nanargmax(sharps))
        return wlist[idx], float(rets[idx]), float(vols[idx]), float(sharps[idx]), False

def plot_frontier_and_cml(vols, rets, sharps, opt_vol, opt_ret, opt_sharpe, rf_ann, labels):
    plt.scatter(vols, rets, c=sharps, cmap="viridis", marker="o")
    cbar = plt.colorbar(); cbar.set_label("Ratio de Sharpe")
    plt.scatter([opt_vol], [opt_ret], c="red", marker="*", s=220, label="Portefeuille optimal")

    # CML (droite de marché des capitaux)
    x = np.linspace(0, max(vols.max(), opt_vol)*1.05, 200)
    y = rf_ann + opt_sharpe * x
    plt.plot(x, y, label="CML", linewidth=2)

    plt.xlabel("Volatilité annuelle")
    plt.ylabel("Rendement annuel")
    plt.title(f"Frontière efficiente (tickers: {labels[0]} & {labels[1]})")
    plt.legend()
    plt.grid(True); plt.tight_layout(); plt.show()

# ============
# 4) Programme
# ============
def main():
    print_header("Analyse interactive — Portefeuille 2 actifs")
    list_universe()
    i1, i2 = ask_two_indices()
    (t1, n1) = UNIVERSE[i1]
    (t2, n2) = UNIVERSE[i2]
    print(f"\nChoix : [{i1}] {t1} — {n1}  &  [{i2}] {t2} — {n2}")

    rf_ann = ask_risk_free()
    start, end = ask_dates()

    print_header("Téléchargement des prix (Adj Close)")
    try:
        prices = fetch_prices([t1, t2], start, end)
    except Exception as e:
        print(f"❌ Échec du téléchargement/préparation des prix : {e}")
        sys.exit(1)

    print("\nAperçu des prix :")
    print(prices.head().to_string(float_format=lambda x: f"{x:0.2f}"))
    prices.to_csv("prices_two_assets.csv", index=True)
    print("💾 Sauvegardé : prices_two_assets.csv")

    # Rendements log
    log_returns = compute_log_returns(prices)
    print_header("Aperçu — rendements log")
    print(log_returns.head().to_string(float_format=lambda x: f"{x:0.6f}"))

    # Statistiques & matrices annualisées
    stats, cov_a, mean_a = describe_returns(log_returns)

    # Courbe des rendements log
    plot_log_returns(log_returns)

    # Volatilités et corrélation glissantes
    rolling_vol_corr(log_returns, window=30)

    # Frontière efficiente (échantillonnage)
    print_header("Frontière efficiente (échantillonnage aléatoire)")
    vols, rets, sharps, wlist = sample_efficient_frontier(mean_a, cov_a, rf_ann, n=4000)

    # Portefeuille de Sharpe max
    w_opt, ret_opt, vol_opt, sh_opt, exact = optimal_sharpe(mean_a, cov_a, rf_ann)
    method_txt = "optimisation (SciPy)" if exact else "approximation (sampling)"
    print(f"Portefeuille max Sharpe — méthode : {method_txt}")
    labels = list(log_returns.columns)
    w_df = pd.Series(w_opt, index=labels, name="Poids")
    print(w_df.to_string(float_format=lambda x: f"{x:0.4f}"))
    print(f"Rendement annuel : {ret_opt:0.4%} | Volatilité annuelle : {vol_opt:0.4%} | Sharpe : {sh_opt:0.4f}")

    # Tracé Frontière + CML
    plot_frontier_and_cml(vols, rets, sharps, vol_opt, ret_opt, sh_opt, rf_ann, labels)

    

if __name__ == "__main__":
    # Quelques options Matplotlib utiles en terminal
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["axes.grid"] = True
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur.")
