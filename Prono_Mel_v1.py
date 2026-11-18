import os
import io
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from scipy.optimize import brentq, minimize
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prono Mel v1 - Calcio AI & Tracker", page_icon="⚽", layout="wide")

QUOTA_MIN = 1.35
QUOTA_MAX = 4.00
MAX_ESPOSIZIONE_GIORNO_PCT = 0.02
FILE_CACHE_STORICO = "cache_prono_mel_storico.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CAMPIONATI = {
    'I1': ('Serie A (Italia)', 'Tier 1 - Top 5 (xG)', 'soccer_italy_serie_a'),
    'E0': ('Premier League (Inghilterra)', 'Tier 1 - Top 5 (xG)', 'soccer_epl'),
    'SP1': ('La Liga (Spagna)', 'Tier 1 - Top 5 (xG)', 'soccer_spain_la_liga'),
    'D1': ('Bundesliga (Germania)', 'Tier 1 - Top 5 (xG)', 'soccer_germany_bundesliga'),
    'F1': ('Ligue 1 (Francia)', 'Tier 1 - Top 5 (xG)', 'soccer_france_ligue_one'),
    'I2': ('Serie B (Italia)', 'Tier 2 - Cadetti', 'soccer_italy_serie_b'),
    'E1': ('Championship (Inghilterra)', 'Tier 2 - Cadetti', 'soccer_efl_champ'),
    'SP2': ('Segunda Division (Spagna)', 'Tier 2 - Cadetti', 'soccer_spain_segunda_division'),
    'D2': ('2. Bundesliga (Germania)', 'Tier 2 - Cadetti', 'soccer_germany_bundesliga2'),
    'F2': ('Ligue 2 (Francia)', 'Tier 2 - Cadetti', 'soccer_france_ligue_two'),
    'N1': ('Eredivisie (Olanda)', 'Tier 3 - Altre Leghe', 'soccer_netherlands_eredivisie'),
    'P1': ('Primeira Liga (Portogallo)', 'Tier 3 - Altre Leghe', 'soccer_portugal_primeira_liga'),
    'B1': ('Pro League (Belgio)', 'Tier 3 - Altre Leghe', 'soccer_belgium_first_div'),
    'T1': ('Super Lig (Turchia)', 'Tier 3 - Altre Leghe', 'soccer_turkey_super_league'),
    'G1': ('Super League (Grecia)', 'Tier 3 - Altre Leghe', 'soccer_greece_super_league')
}

def devig_power_singolo(quote):
    q = np.asarray(quote, dtype=float)
    if np.any(np.isnan(q)) or np.any(q <= 1.0):
        return np.full(len(q), np.nan)
    inv = 1.0 / q
    if np.sum(inv) <= 1.0:
        return inv / np.sum(inv)
    try:
        c = brentq(lambda x: np.sum(inv ** x) - 1.0, 1.0, 10.0)
        return inv ** c
    except Exception:
        return inv / np.sum(inv)

def devig_power_colonne(df, cols):
    inv = 1.0 / df[cols].values.astype(float)
    invalid = np.any(np.isnan(inv), axis=1) | np.any(inv >= 1.0, axis=1)
    c = np.ones(len(inv))
    for _ in range(15):
        p = inv ** c[:, None]
        f = np.sum(p, axis=1) - 1.0
        df_dc = np.sum(p * np.log(np.clip(inv, 1e-12, 1.0)), axis=1)
        c = np.where(invalid | (np.abs(df_dc) < 1e-12), c, c - f / df_dc)
    res = inv ** c[:, None]
    res[invalid] = np.nan
    return res
def rinomina_colonne_csv(df):
    return df.rename(columns={
        'B365>2.5': 'B365_Over25', 'B365<2.5': 'B365_Under25',
        'Max>2.5': 'Max_Over25', 'Max<2.5': 'Max_Under25',
        'Avg>2.5': 'Avg_Over25', 'Avg<2.5': 'Avg_Under25',
        'AvgC>2.5': 'AvgC_Over25', 'AvgC<2.5': 'AvgC_Under25'
    })

def scarica_csv_veloce(url, tentativi=2):
    for t in range(tentativi):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            if "<html" in r.text.lower():
                return None
            return r.text
        except Exception:
            time.sleep(0.5 * (t + 1))
    return None

def scarica_stagioni(lista_stagioni):
    fogli, falliti = [], []
    colonne_extra = [
        'HS', 'AS', 'HST', 'AST', 'B365_Over25', 'B365_Under25',
        'AvgH', 'AvgD', 'AvgA', 'Avg_Over25', 'Avg_Under25',
        'MaxH', 'MaxD', 'MaxA', 'Max_Over25', 'Max_Under25',
        'AvgCH', 'AvgCD', 'AvgCA', 'AvgC_Over25', 'AvgC_Under25'
    ]
    for cod in CAMPIONATI.keys():
        for s in lista_stagioni:
            url = f"https://football-data.co.uk{s}/{cod}.csv"
            txt = scarica_csv_veloce(url, tentativi=2)
            if txt is None:
                falliti.append(f"{cod}_{s}")
                continue
            try:
                df = pd.read_csv(io.StringIO(txt), on_bad_lines='skip')
                df = rinomina_colonne_csv(df)
                base = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
                if not all(c in df.columns for c in base):
                    continue
                for col_ex in colonne_extra:
                    if col_ex not in df.columns:
                        df[col_ex] = np.nan
                df = df[base + colonne_extra].copy()
                df['Div'] = cod
                fogli.append(df)
            except Exception:
                falliti.append(f"{cod}_{s}")
    return (pd.concat(fogli, ignore_index=True) if fogli else pd.DataFrame()), falliti

def cerca_migliore_quota_live(api_key, sport_key, casa, trasferta, mercato):
    if not api_key:
        return None, "Inserisci una API Key gratuita di The Odds API nella barra laterale."
    mkt_api = "totals" if "2.5" in mercato else "h2h"
    url = f"https://the-odds-api.com{sport_key}/odds/?apiKey={api_key}&regions=eu,uk&markets={mkt_api}&oddsFormat=decimal"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None, f"Errore API ({r.status_code}): controlla la chiave."
        dati = r.json()
        casa_low, trasf_low = casa.lower()[:5], trasferta.lower()[:5]
        for evento in dati:
            h_ev = evento.get('home_team', '').lower()
            a_ev = evento.get('away_team', '').lower()
            if casa_low in h_ev or trasf_low in a_ev:
                offerte = []
                for bk in evento.get('bookmakers', []):
                    nome_bk = bk.get('title', 'Bookmaker')
                    for m in bk.get('markets', []):
                        for out in m.get('outcomes', []):
                            nome_out = out.get('name', '')
                            prezzo = float(out.get('price', 0.0))
                            punto = out.get('point', None)
                            match_ok = False
                            if mercato == '1' and nome_out == evento.get('home_team'):
                                match_ok = True
                            elif mercato == '2' and nome_out == evento.get('away_team'):
                                match_ok = True
                            elif mercato == 'X' and nome_out.lower() == 'draw':
                                match_ok = True
                            elif mercato == 'OVER 2.5' and nome_out.lower() == 'over' and punto == 2.5:
                                match_ok = True
                            elif mercato == 'UNDER 2.5' and nome_out.lower() == 'under' and punto == 2.5:
                                match_ok = True
                            if match_ok and prezzo > 1.0:
                                offerte.append({'Bookmaker': nome_bk, 'Quota Live': prezzo})
                if offerte:
                    return pd.DataFrame(offerte).sort_values(by='Quota Live', ascending=False).reset_index(drop=True), "OK"
        return None, "Partita non ancora quotata su The Odds API."
    except Exception as e:
        return None, f"Errore connessione: {e}"

def get_gsheets_conn():
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception:
        return None

def carica_tracker():
    colonne = [
        'ID', 'Data_Match', 'Campionato', 'Tier', 'Partita', 'Mercato',
        'Variante_Modello', 'Bookmaker', 'Quota_Presa', 'Quota_Chiusura',
        'Prob_AI_Pct', 'Malus_Assenze_Pct', 'Puntata_Euro', 'Esito', 'Profitto_Euro', 'CLV_Netto_Pct'
    ]
    conn = get_gsheets_conn()
    if conn is not None:
        try:
            df = conn.read(worksheet="Tracker", ttl=0)
            if df is not None and not df.empty:
                return df.dropna(how='all').reset_index(drop=True)
        except Exception:
            pass
    if os.path.exists("prono_mel_tracker.csv"):
        return pd.read_csv("prono_mel_tracker.csv")
    return pd.DataFrame(columns=colonne)

def salva_tracker(df_tr):
    df_tr.to_csv("prono_mel_tracker.csv", index=False)
    conn = get_gsheets_conn()
    if conn is not None:
        try:
            conn.update(worksheet="Tracker", data=df_tr)
        except Exception:
            pass

def registra_scarto(riga_dict):
    df_sc = pd.DataFrame([riga_dict])
    conn = get_gsheets_conn()
    if conn is not None:
        try:
            esistente = conn.read(worksheet="Scarti", ttl=0)
            if esistente is not None and not esistente.empty:
                df_sc = pd.concat([esistente.dropna(how='all'), df_sc], ignore_index=True)
            conn.update(worksheet="Scarti", data=df_sc)
            return
        except Exception:
            pass
    if os.path.exists("prono_mel_scarti.csv"):
        df_sc = pd.concat([pd.read_csv("prono_mel_scarti.csv"), df_sc], ignore_index=True)
    df_sc.to_csv("prono_mel_scarti.csv", index=False)
@st.cache_resource(show_spinner="Caricamento storico e addestramento rapido Prono Mel v1...")
def prepara_motore_predittivo(forza_aggiornamento=False):
    oggi_dt = pd.Timestamp.now()
    anno_corr = oggi_dt.year if oggi_dt.month >= 7 else oggi_dt.year - 1
    stagione_corr = f"{str(anno_corr)[-2:]}{str(anno_corr + 1)[-2:]}"
    stagioni_pass = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(2019, anno_corr)]

    if os.path.exists(FILE_CACHE_STORICO) and not forza_aggiornamento:
        df_pass = pd.read_csv(FILE_CACHE_STORICO)
    else:
        df_pass, falliti = scarica_stagioni(stagioni_pass)
        if not falliti and not df_pass.empty:
            df_pass.to_csv(FILE_CACHE_STORICO, index=False)

    df_curr, _ = scarica_stagioni([stagione_corr])
    db = pd.concat([df_pass, df_curr], ignore_index=True)
    db = db.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A'])
    db['Data_DT'] = pd.to_datetime(db['Date'], dayfirst=True, format='mixed', errors='coerce')
    db = db.dropna(subset=['Data_DT']).sort_values(['Data_DT', 'Div', 'HomeTeam']).reset_index(drop=True)

    for c_alt, c_b365 in [
        ('MaxH', 'B365H'), ('MaxD', 'B365D'), ('MaxA', 'B365A'),
        ('Max_Over25', 'B365_Over25'), ('Max_Under25', 'B365_Under25'),
        ('AvgH', 'B365H'), ('AvgD', 'B365D'), ('AvgA', 'B365A'),
        ('Avg_Over25', 'B365_Over25'), ('Avg_Under25', 'B365_Under25')
    ]:
        db[c_alt] = db[c_alt].fillna(db[c_b365])

    elo_ratings, ultima_data, ultima_div = {}, {}, {}
    promozioni = {'I2': 'I1', 'E1': 'E0', 'SP2': 'SP1', 'D2': 'D1', 'F2': 'F1'}
    h_elo, a_elo, h_rest, a_rest = [], [], [], []

    for row in db[['Data_DT', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].itertuples(index=False):
        dt_c, div_c, ht, at, hg, ag = row
        rh, ra = elo_ratings.get(ht, 1500.0), elo_ratings.get(at, 1500.0)
        if ht in ultima_data:
            dh = (dt_c - ultima_data[ht]).days
            if dh > 45:
                rh = 0.75 * rh + 0.25 * 1500.0
                if promozioni.get(ultima_div.get(ht)) == div_c:
                    rh -= 50.0
            h_rest.append(min(max(dh, 1), 14))
        else:
            h_rest.append(7)
        if at in ultima_data:
            da = (dt_c - ultima_data[at]).days
            if da > 45:
                ra = 0.75 * ra + 0.25 * 1500.0
                if promozioni.get(ultima_div.get(at)) == div_c:
                    ra -= 50.0
            a_rest.append(min(max(da, 1), 14))
        else:
            a_rest.append(7)
        h_elo.append(rh); a_elo.append(ra)
        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + 65.0)) / 400.0))
        res_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo_ratings[ht] = rh + 20.0 * (res_h - exp_h)
        elo_ratings[at] = ra + 20.0 * ((1.0 - res_h) - (1.0 - exp_h))
        ultima_data[ht], ultima_data[at] = dt_c, dt_c
        ultima_div[ht], ultima_div[at] = div_c, div_c
    db['Elo_Diff'] = np.array(h_elo) - np.array(a_elo)
    db['Abs_Elo_Diff'] = db['Elo_Diff'].abs()
    db['Rest_Diff'] = np.array(h_rest) - np.array(a_rest)
    db['TotalGoals'] = db['FTHG'] + db['FTAG']
    db['Lega_Gol'] = db.groupby('Div')['TotalGoals'].transform(lambda x: x.shift(1).rolling(200, min_periods=20).mean())

    p_open_1x2 = devig_power_colonne(db, ['AvgH', 'AvgD', 'AvgA'])
    db['Mkt_P1'], db['Mkt_PX'], db['Mkt_P2'] = p_open_1x2[:, 0], p_open_1x2[:, 1], p_open_1x2[:, 2]
    p_open_ou = devig_power_colonne(db, ['Avg_Over25', 'Avg_Under25'])
    db['Mkt_PO25'], db['Mkt_PU25'] = p_open_ou[:, 0], p_open_ou[:, 1]

    db['H_Pts'] = np.where(db['FTHG'] > db['FTAG'], 3, np.where(db['FTHG'] == db['FTAG'], 1, 0))
    db['A_Pts'] = np.where(db['FTAG'] > db['FTHG'], 3, np.where(db['FTAG'] == db['FTHG'], 1, 0))
    db['H_Res_Pts'] = db['H_Pts'] - (3.0 * db['Mkt_P1'] + 1.0 * db['Mkt_PX'])
    db['A_Res_Pts'] = db['A_Pts'] - (3.0 * db['Mkt_P2'] + 1.0 * db['Mkt_PX'])
    db['H_SOT_Diff'] = db['HST'] - db['AST']
    db['A_SOT_Diff'] = db['AST'] - db['HST']

    db['H_EWM_ResPts'] = db.groupby('HomeTeam')['H_Res_Pts'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['A_EWM_ResPts'] = db.groupby('AwayTeam')['A_Res_Pts'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['H_EWM_SOT_Diff'] = db.groupby('HomeTeam')['H_SOT_Diff'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['A_EWM_SOT_Diff'] = db.groupby('AwayTeam')['A_SOT_Diff'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['H_EWM_GF'] = db.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['A_EWM_GF'] = db.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())

    db['Target_1'] = (db['FTHG'] > db['FTAG']).astype(int)
    db['Target_X'] = (db['FTHG'] == db['FTAG']).astype(int)
    db['Target_2'] = (db['FTHG'] < db['FTAG']).astype(int)
    db['Target_Over25'] = (db['TotalGoals'] > 2.5).astype(int)
    db['Target_Under25'] = (db['TotalGoals'] < 2.5).astype(int)

    feat_cols = [
        'H_EWM_ResPts', 'A_EWM_ResPts', 'H_EWM_SOT_Diff', 'A_EWM_SOT_Diff',
        'H_EWM_GF', 'A_EWM_GF', 'Elo_Diff', 'Abs_Elo_Diff', 'Rest_Diff', 'Lega_Gol'
    ]
    medie_feat = db[feat_cols].mean()
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(db[feat_cols].fillna(medie_feat))

    eps = 1e-5
    modelli = {}
    mercati_cfg = {
        '1': ('Target_1', 'Mkt_P1'), 'X': ('Target_X', 'Mkt_PX'), '2': ('Target_2', 'Mkt_P2'),
        'OVER 2.5': ('Target_Over25', 'Mkt_PO25'), 'UNDER 2.5': ('Target_Under25', 'Mkt_PU25')
    }
    for m_nome, (col_y, col_mkt) in mercati_cfg.items():
        valid = db[col_mkt].notna().values
        y_v = db.loc[valid, col_y].values
        off_v = logit(np.clip(db.loc[valid, col_mkt].values, eps, 1 - eps))
        X_v = X_all_s[valid]
        def loss_grad(params):
            b0, b = float(params.item(0)), params[1:]
            p = expit(off_v + b0 + X_v @ b)
            ll = -np.mean(y_v * np.log(np.clip(p, eps, 1 - eps)) + (1 - y_v) * np.log(np.clip(1 - p, eps, 1 - eps)))
            reg = 0.5 * (400.0 / len(y_v)) * np.sum(b ** 2)
            diff = (p - y_v) / len(y_v)
            return ll + reg, np.concatenate([[np.sum(diff)], X_v.T @ diff + (400.0 / len(y_v)) * b])
        res = minimize(loss_grad, np.zeros(len(feat_cols) + 1), jac=True, method='BFGS')
        modelli[m_nome] = (float(res.x.item(0)), res.x[1:])

    return db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli
def analizza_palinsesto_futuro(db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli, soglia_ev, flat_stake, budget_tot, solo_top5=True):
    txt_fix = scarica_csv_veloce("https://football-data.co.uk", tentativi=2)
    if txt_fix is None:
        return pd.DataFrame(), "Impossibile scaricare fixtures.csv al momento."
    df_fix = rinomina_colonne_csv(pd.read_csv(io.StringIO(txt_fix), on_bad_lines='skip'))
    if 'Div' not in df_fix.columns:
        return pd.DataFrame(), "Palinsesto non disponibile."

    df_fix = df_fix[df_fix['Div'].isin(list(CAMPIONATI.keys()))].dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']).copy()
    df_fix['Data_DT'] = pd.to_datetime(df_fix['Date'], dayfirst=True, format='mixed', errors='coerce')
    oggi = pd.Timestamp.now().normalize()
    df_fix = df_fix[df_fix['Data_DT'] >= oggi].copy()

    if solo_top5:
        df_fix = df_fix[df_fix['Div'].isin(['I1', 'E0', 'SP1', 'D1', 'F1'])].copy()
    if df_fix.empty:
        return pd.DataFrame(), "Nessuna partita futura in programma per i campionati selezionati."

    promozioni = {'I2': 'I1', 'E1': 'E0', 'SP2': 'SP1', 'D2': 'D1', 'F2': 'F1'}
    eps = 1e-5
    segnali = []

    for _, row in df_fix.iterrows():
        cod = row['Div']
        nome_lega, tier_lega, sport_key = CAMPIONATI[cod]
        casa, trasf, data_str, dt_m = row['HomeTeam'], row['AwayTeam'], str(row['Date']), row['Data_DT']

        q1, qX, q2 = float(row['B365H']), float(row['B365D']), float(row['B365A'])
        avg1 = float(row['AvgH']) if pd.notna(row.get('AvgH')) else q1
        avgX = float(row['AvgD']) if pd.notna(row.get('AvgD')) else qX
        avg2 = float(row['AvgA']) if pd.notna(row.get('AvgA')) else q2
        max1 = float(row['MaxH']) if pd.notna(row.get('MaxH')) else q1
        maxX = float(row['MaxD']) if pd.notna(row.get('MaxD')) else qX
        max2 = float(row['MaxA']) if pd.notna(row.get('MaxA')) else q2

        qo = float(row['B365_Over25']) if pd.notna(row.get('B365_Over25')) else np.nan
        qu = float(row['B365_Under25']) if pd.notna(row.get('B365_Under25')) else np.nan
        avgo = float(row['Avg_Over25']) if pd.notna(row.get('Avg_Over25')) else qo
        avgu = float(row['Avg_Under25']) if pd.notna(row.get('Avg_Under25')) else qu
        maxo = float(row['Max_Over25']) if pd.notna(row.get('Max_Over25')) else qo
        maxu = float(row['Max_Under25']) if pd.notna(row.get('Max_Under25')) else qu

        rh, ra = elo_ratings.get(casa, 1500.0), elo_ratings.get(trasf, 1500.0)
        if casa in ultima_data:
            dh = (dt_m - ultima_data[casa]).days
            if dh > 45:
                rh = 0.75 * rh + 0.25 * 1500.0
                if promozioni.get(ultima_div.get(casa)) == cod:
                    rh -= 50.0
            rest_h = min(max(dh, 1), 14)
        else:
            rest_h = 7
        if trasf in ultima_data:
            da = (dt_m - ultima_data[trasf]).days
            if da > 45:
                ra = 0.75 * ra + 0.25 * 1500.0
                if promozioni.get(ultima_div.get(trasf)) == cod:
                    ra -= 50.0
            rest_a = min(max(da, 1), 14)
        else:
            rest_a = 7

        st_val = db[db['Data_DT'] < dt_m]
        lega_mg = st_val[st_val['Div'] == cod]['TotalGoals'].tail(200).mean()
        if pd.isna(lega_mg):
            lega_mg = medie_feat['Lega_Gol']

        pc = st_val[st_val['HomeTeam'] == casa]
        pt = st_val[st_val['AwayTeam'] == trasf]

        row_x = pd.DataFrame([{
            'H_EWM_ResPts': pc['H_Res_Pts'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pc) > 0 else medie_feat['H_EWM_ResPts'],
            'A_EWM_ResPts': pt['A_Res_Pts'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pt) > 0 else medie_feat['A_EWM_ResPts'],
            'H_EWM_SOT_Diff': pc['H_SOT_Diff'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pc) > 0 else medie_feat['H_EWM_SOT_Diff'],
            'A_EWM_SOT_Diff': pt['A_SOT_Diff'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pt) > 0 else medie_feat['A_EWM_SOT_Diff'],
            'H_EWM_GF': pc['FTHG'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pc) > 0 else medie_feat['H_EWM_GF'],
            'A_EWM_GF': pt['FTAG'].ewm(span=10, min_periods=1).mean().iloc[-1] if len(pt) > 0 else medie_feat['A_EWM_GF'],
            'Elo_Diff': rh - ra, 'Abs_Elo_Diff': abs(rh - ra),
            'Rest_Diff': rest_h - rest_a, 'Lega_Gol': lega_mg
        }]).fillna(medie_feat)

        X_s = scaler.transform(row_x)
        p_mkt_1x2 = devig_power_singolo([avg1, avgX, avg2])
        p_raw = {}
        for idx_m, m_nome in enumerate(['1', 'X', '2']):
            b0, b = modelli[m_nome]
            off = float(logit(np.clip(p_mkt_1x2.item(idx_m), eps, 1 - eps)))
            p_raw[m_nome] = float(expit(off + b0 + (X_s @ b).item(0)))
        s1x2 = p_raw['1'] + p_raw['X'] + p_raw['2']
        p_fin = {'1': p_raw['1']/s1x2, 'X': p_raw['X']/s1x2, '2': p_raw['2']/s1x2}

        if pd.notna(avgo) and pd.notna(avgu) and avgo > 1.0 and avgu > 1.0:
            p_mkt_ou = devig_power_singolo([avgo, avgu])
            for idx_o, m_nome in enumerate(['OVER 2.5', 'UNDER 2.5']):
                b0, b = modelli[m_nome]
                off = float(logit(np.clip(p_mkt_ou.item(idx_o), eps, 1 - eps)))
                p_raw[m_nome] = float(expit(off + b0 + (X_s @ b).item(0)))
            sou = p_raw['OVER 2.5'] + p_raw['UNDER 2.5']
            p_fin['OVER 2.5'], p_fin['UNDER 2.5'] = p_raw['OVER 2.5']/sou, p_raw['UNDER 2.5']/sou

        mappa_quote = {
            '1': (q1, avg1, max1), 'X': (qX, avgX, maxX), '2': (q2, avg2, max2),
            'OVER 2.5': (qo, avgo, maxo), 'UNDER 2.5': (qu, avgu, maxu)
        }
        for m_nome, p_ai in p_fin.items():
            qb, qavg, qmax = mappa_quote[m_nome]
            q_rif = qmax if pd.notna(qmax) else qb
            if pd.notna(q_rif) and QUOTA_MIN <= q_rif <= QUOTA_MAX:
                ev = (p_ai * q_rif) - 1.0
                if ev >= soglia_ev:
                    segnali.append({
                        'Data': data_str, 'Campionato': nome_lega, 'Tier': tier_lega,
                        'Sport_Key': sport_key, 'Casa': casa, 'Trasferta': trasf,
                        'Partita': f"{casa} - {trasf}", 'Mercato': m_nome,
                        'Quota_B365': qb, 'Quota_Media': round(qavg, 2), 'Quota_MAX': qmax,
                        'Prob_AI_Pct': round(p_ai * 100.0, 2),
                        'EV_Pct': round(ev * 100.0, 2),
                        'Puntata_Consigliata': flat_stake
                    })

    if not segnali:
        return pd.DataFrame(), "Nessun segnale supera i filtri di sicurezza (EV >= soglia e quota 1.35-4.00)."

    df_seg = pd.DataFrame(segnali).sort_values(by='EV_Pct', ascending=False).reset_index(drop=True)
    cap_euro = budget_tot * MAX_ESPOSIZIONE_GIORNO_PCT
    for dt_v, grp in df_seg.groupby('Data'):
        tot_d = grp['Puntata_Consigliata'].sum()
        if tot_d > cap_euro and tot_d > 0:
            df_seg.loc[grp.index, 'Puntata_Consigliata'] = (df_seg.loc[grp.index, 'Puntata_Consigliata'] * (cap_euro / tot_d)).round(2)

    return df_seg, "OK"
st.title("⚽ Prono Mel v1 — Motore Quantitativo & Profit Tracker")
st.caption("Cloud Edition: Power Devig + Logit-Offset + Outcome Bias + Sincronizzazione Google Sheets")

with st.sidebar:
    st.header("⚙️ Parametri Operativi")
    budget_totale = st.number_input("Capitale di Riferimento (€)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
    modalita_operativa = st.radio(
        "Modalità Staking",
        ["Paper Trading (1% = 10€)", "Pilota Reale Micro (0.5% = 5€)"],
        index=0
    )
    flat_stake = round(budget_totale * (0.01 if "Paper" in modalita_operativa else 0.005), 2)
    st.info(f"Puntata fissa attiva: **{flat_stake:.2f} €** | Cap giornaliero (2%): **{budget_totale * 0.02:.2f} €**")

    soglia_ev_pct = st.slider("Soglia Minima EV (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    solo_top5 = st.checkbox("Concentra sui Top 5 Campionati (Consigliato)", value=True)

    st.divider()
    st.subheader("🔑 Line Shopping Live (Opzionale)")
    odds_api_key = st.text_input("API Key The Odds API (500 req/mese gratis)", type="password")

    if st.button("🔄 Aggiorna Archivio Storico"):
        st.cache_resource.clear()
        st.rerun()

db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli = prepara_motore_predittivo()
st.sidebar.success(f"Database attivo: {len(db)} partite analizzate")

tab_prono, tab_tracker, tab_guida = st.tabs([
    "🎯 1. Pronostici & Line Shopping",
    "📊 2. Profit Tracker & Semaforo",
    "📘 3. Routine & Regole Operative"
])

with tab_prono:
    st.subheader("Analisi Palinsesto Settimanale")
    if st.button("🚀 Analizza Partite Future", type="primary"):
        df_seg, msg = analizza_palinsesto_futuro(
            db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli,
            soglia_ev=soglia_ev_pct / 100.0, flat_stake=flat_stake, budget_tot=budget_totale, solo_top5=solo_top5
        )
        st.session_state['segnali_correnti'] = df_seg
        st.session_state['msg_segnali'] = msg

    if 'segnali_correnti' in st.session_state:
        df_seg = st.session_state['segnali_correnti']
        if df_seg.empty:
            st.warning(st.session_state.get('msg_segnali', 'Nessun segnale trovato.'))
        else:
            st.dataframe(
                df_seg[['Data', 'Campionato', 'Partita', 'Mercato', 'Quota_B365', 'Quota_Media', 'Quota_MAX', 'Prob_AI_Pct', 'EV_Pct', 'Puntata_Consigliata']],
                use_container_width=True
            )

            st.divider()
            st.subheader("🔍 Verifica Quota Live, Malus Assenze (T-45 min) e Registrazione")
            opzioni_match = [f"{i} | {r['Partita']} [{r['Mercato']}] @ {r['Quota_MAX']}" for i, r in df_seg.iterrows()]
            scelta_idx = st.selectbox("Seleziona la partita da verificare:", range(len(opzioni_match)), format_func=lambda x: opzioni_match[x])
            riga = df_seg.iloc[scelta_idx]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**A. Comparatore Quote Live (The Odds API)**")
                if st.button("🔎 Cerca Quota Migliore tra i Bookmaker"):
                    df_live, esito_api = cerca_migliore_quota_live(odds_api_key, riga['Sport_Key'], riga['Casa'], riga['Trasferta'], riga['Mercato'])
                    if df_live is not None:
                        st.dataframe(df_live, use_container_width=True)
                    else:
                        st.info(esito_api)

            with col_b:
                st.markdown("**B. Check Assenze & Calcolo EV Finale**")
                malus_pct = st.slider("Malus Assenze / Infortuni (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
                prob_corretta = max(1.0, min(99.0, riga['Prob_AI_Pct'] + malus_pct))
                book_scelto = st.text_input("Bookmaker / Exchange scelto", value="Miglior Quota ADM / Exchange")
                quota_reale = st.number_input("Quota reale trovata", min_value=1.01, max_value=20.0, value=float(riga['Quota_MAX']), step=0.01)
                ev_finale = ((prob_corretta / 100.0) * quota_reale - 1.0) * 100.0

                variante = "Modello Base Congelato" if malus_pct == 0.0 else "Intervento Manuale (Assenze)"
                if ev_finale >= soglia_ev_pct:
                    st.success(f"Probabilità Finale: **{prob_corretta:.1f}%** | EV Reale: **+{ev_finale:.2f}%** (Giocata Valida)")
                else:
                    st.error(f"Probabilità Finale: **{prob_corretta:.1f}%** | EV Reale: **{ev_finale:.2f}%** (Sotto soglia: scartare)")

                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    if st.button("✅ Registra nel Tracker", type="primary"):
                        df_tr = carica_tracker()
                        nuovo_id = int(pd.to_numeric(df_tr['ID'], errors='coerce').max() + 1) if not df_tr.empty else 1
                        nuova_riga = {
                            'ID': nuovo_id, 'Data_Match': riga['Data'], 'Campionato': riga['Campionato'],
                            'Tier': riga['Tier'], 'Partita': riga['Partita'], 'Mercato': riga['Mercato'],
                            'Variante_Modello': variante, 'Bookmaker': book_scelto,
                            'Quota_Presa': quota_reale, 'Quota_Chiusura': np.nan,
                            'Prob_AI_Pct': prob_corretta, 'Malus_Assenze_Pct': malus_pct,
                            'Puntata_Euro': riga['Puntata_Consigliata'], 'Esito': 'In Corso',
                            'Profitto_Euro': 0.0, 'CLV_Netto_Pct': np.nan
                        }
                        df_tr = pd.concat([df_tr, pd.DataFrame([nuova_riga])], ignore_index=True)
                        salva_tracker(df_tr)
                        st.success("Giocata salvata sul Foglio Google!")

                with c_btn2:
                    motivo_scarto = st.text_input("Motivo eventuale scarto", value="Quota scesa / Assenze chiave")
                    if st.button("🗑️ Scarta e Archivia Decisione"):
                        registra_scarto({
                            'Data_Log': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                            'Partita': riga['Partita'], 'Mercato': riga['Mercato'],
                            'Quota_Rif': quota_reale, 'EV_Finale_Pct': round(ev_finale, 2),
                            'Motivo': motivo_scarto
                        })
                        st.info("Decisione di scarto archiviata su Google Sheets.")
with tab_tracker:
    st.subheader("📊 Profit Tracker & Semaforo Validazione")
    df_tr = carica_tracker()

    if df_tr.empty:
        st.info("Nessuna giocata ancora registrata. Usa la Scheda 1 per salvare le tue prime giocate in Paper Trading.")
    else:
        filtro_var = st.selectbox("Filtra per Variante Modello (Separazione A/B/C):", ["Tutte"] + sorted(df_tr['Variante_Modello'].dropna().unique().tolist()))
        df_view = df_tr if filtro_var == "Tutte" else df_tr[df_tr['Variante_Modello'] == filtro_var].copy()

        chiuse = df_view[df_view['Esito'].isin(['Vinta', 'Persa'])].copy()
        n_chiuse = len(chiuse)
        prof_tot = float(pd.to_numeric(chiuse['Profitto_Euro'], errors='coerce').sum()) if n_chiuse > 0 else 0.0
        vol_tot = float(pd.to_numeric(chiuse['Puntata_Euro'], errors='coerce').sum()) if n_chiuse > 0 else 0.0
        roi_pct = (prof_tot / vol_tot * 100.0) if vol_tot > 0 else 0.0
        clv_s = pd.to_numeric(chiuse['CLV_Netto_Pct'], errors='coerce').dropna()
        clv_medio = float(clv_s.mean()) if not clv_s.empty else 0.0

        equity = budget_totale + pd.to_numeric(chiuse['Profitto_Euro'], errors='coerce').fillna(0.0).cumsum()
        picco = np.maximum.accumulate(np.concatenate([[budget_totale], equity.values]))
        dd_pct = float(np.max((picco - np.concatenate([[budget_totale], equity.values])) / picco) * 100.0) if n_chiuse > 0 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Giocate Chiuse", f"{n_chiuse} / 300")
        m2.metric("Profitto Netto", f"{prof_tot:+.2f} €")
        m3.metric("ROI Reale", f"{roi_pct:+.2f}%")
        m4.metric("CLV Netto Medio", f"{clv_medio:+.2f}%")
        m5.metric("Max Drawdown", f"{dd_pct:.2f}%")

        st.progress(min(1.0, n_chiuse / 300.0), text=f"Avanzamento Validazione Paper Trading: {n_chiuse} su 300 giocate minime")

        if dd_pct >= 15.0:
            st.error("🚨 STOP-LOSS SCATTATO: Drawdown >= 15%. Sospendere le giocate e rivedere i parametri!")
        elif n_chiuse >= 300 and clv_medio > 0 and dd_pct < 15.0:
            st.success("🟢 SEMAFORO VERDE: Criteri di promozione raggiunti! Puoi avviare il Pilota Reale Micro (0.5%).")
        else:
            st.warning("🟡 SEMAFORO GIALLO: Continua il Paper Trading fino a 300 giocate con CLV netto positivo.")

        if n_chiuse > 1:
            st.line_chart(pd.DataFrame({'Capitale (€)': equity.values}, index=range(1, n_chiuse + 1)))

        st.divider()
        st.markdown("### ✏️ Aggiorna Esiti e Quota di Chiusura")
        in_corso = df_tr[df_tr['Esito'] == 'In Corso']
        if not in_corso.empty:
            id_sel = st.selectbox("Seleziona ID giocata da chiudere:", in_corso['ID'].tolist())
            r_sel = df_tr[df_tr['ID'] == id_sel].iloc.to_dict()
            st.write(f"**{r_sel['Partita']}** — Mercato: **{r_sel['Mercato']}** @ **{r_sel['Quota_Presa']}** (Puntata: {r_sel['Puntata_Euro']} €)")
            c_e1, c_e2, c_e3 = st.columns(3)
            with c_e1:
                nuovo_esito = st.selectbox("Esito Finale", ["Vinta", "Persa", "Rimborsata"])
            with c_e2:
                q_chiusura = st.number_input("Quota Chiusura Reale (al calcio d'inizio)", min_value=1.01, max_value=25.0, value=float(r_sel['Quota_Presa']), step=0.01)
            with c_e3:
                aggio_chiusura = st.number_input("Aggio stimato chiusura (%)", min_value=0.0, max_value=10.0, value=2.5, step=0.5)

            if st.button("💾 Salva Chiusura Giocata"):
                idx_row = df_tr[df_tr['ID'] == id_sel].index.item(0)
                stake_v = float(df_tr.loc[idx_row, 'Puntata_Euro'])
                q_presa = float(df_tr.loc[idx_row, 'Quota_Presa'])
                prof_v = round(stake_v * (q_presa - 1.0), 2) if nuovo_esito == "Vinta" else (-stake_v if nuovo_esito == "Persa" else 0.0)
                p_close_clean = (1.0 / q_chiusura) / (1.0 + aggio_chiusura / 100.0)
                clv_net = round(((q_presa * p_close_clean) - 1.0) * 100.0, 2)

                df_tr.loc[idx_row, 'Esito'] = nuovo_esito
                df_tr.loc[idx_row, 'Quota_Chiusura'] = q_chiusura
                df_tr.loc[idx_row, 'Profitto_Euro'] = prof_v
                df_tr.loc[idx_row, 'CLV_Netto_Pct'] = clv_net
                salva_tracker(df_tr)
                st.success(f"Esito salvato! Profitto: {prof_v:+.2f} € | CLV Netto: {clv_net:+.2f}%")
                st.rerun()

        st.markdown("### 📋 Registro Completo Giocate")
        st.dataframe(df_view.sort_values(by='ID', ascending=False), use_container_width=True)

with tab_guida:
    st.subheader("📘 Routine Settimanale & Regole di Gestione Rischio")
    st.markdown("""
    * **Lunedì Mattina (09:00 - 12:00):** Aggiorna esiti e quote di chiusura nel Profit Tracker e riconcilia il saldo.
    * **Giovedì Pomeriggio (18:00) / Venerdì (10:00 - 19:00):** Lancia l'analisi del weekend, usa il comparatore per il *Line Shopping* e registra le giocate candidate.
    * **Sabato / Domenica (T-45 minuti):** Controlla le formazioni ufficiali e gli infortuni. Usa il cursore *Malus Assenze* solo per gestire o scartare le giocate già individuate se mancano titolari chiave.
    * **Regola d'Oro:** Non superare mai il tetto giornaliero del **2%** e attendi almeno **300 giocate** con **CLV Netto positivo** prima di passare dal Paper Trading al Pilota Reale Micro (0.5%).
    """)
