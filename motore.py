# BLOCCO 1: Importazioni, parametri base, campionati e funzioni Power Devig
import os
import io
import time
import requests
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import logit, expit
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler

QUOTA_MIN = 1.35
QUOTA_MAX = 4.00
MAX_ESPOSIZIONE_GIORNO_PCT = 0.02
FILE_CACHE_STORICO = "cache_prono_mel_storico.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://football-data.co.uk"
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

LEGHE_POISSON = ['I1', 'E0', 'SP1', 'D1']

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


# BLOCCO 2: Pulizia CSV, download stagioni, matrice Poisson Dixon-Coles e CLV chiusura
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
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            if "<html" in r.text.lower():
                return None
            return r.text
        except Exception:
            time.sleep(0.5 * (t + 1))
    return None

def scarica_stagioni(lista_stagioni, codici_leghe=None):
    fogli, falliti = [], []
    leghe = codici_leghe if codici_leghe is not None else list(CAMPIONATI.keys())
    colonne_extra = [
        'HS', 'AS', 'HST', 'AST', 'B365_Over25', 'B365_Under25',
        'AvgH', 'AvgD', 'AvgA', 'Avg_Over25', 'Avg_Under25',
        'MaxH', 'MaxD', 'MaxA', 'Max_Over25', 'Max_Under25',
        'AvgCH', 'AvgCD', 'AvgCA', 'AvgC_Over25', 'AvgC_Under25'
    ]
    for cod in leghe:
        for s in lista_stagioni:
            url = "https://www.football-data.co.uk" + f"/mmz4281/{s}/{cod}.csv"
            txt = scarica_csv_veloce(url, tentativi=2)
            if txt is None:
                falliti.append(f"{cod}_{s}")
                continue
            try:
                df = pd.read_csv(io.StringIO(txt), on_bad_lines='skip', encoding='utf-8-sig', engine='python')
                df.columns = df.columns.str.replace('ï»¿', '')
                df = rinomina_colonne_csv(df)
                base = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']
                if not all(c in df.columns for c in base):
                    continue
                for col_ex in colonne_extra:
                    if col_ex not in df.columns:
                        df[col_ex] = np.nan
                df = df[base + colonne_extra].copy()
                df['Div'] = cod
                df['Stagione'] = s
                fogli.append(df)
            except Exception:
                falliti.append(f"{cod}_{s}")
    return (pd.concat(fogli, ignore_index=True) if fogli else pd.DataFrame()), falliti

def calcola_poisson_dixon_coles(lam_h, lam_a, rho=-0.06):
    k = np.arange(7)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    mat = np.outer(ph, pa)
    mat *= max(0.1, 1.0 - lam_h * lam_a * rho)
    mat *= max(0.1, 1.0 + lam_h * rho)
    mat *= max(0.1, 1.0 + lam_a * rho)
    mat *= max(0.1, 1.0 - rho)
    mat = mat / np.sum(mat)

    p1 = float(np.sum(np.tril(mat, -1)))
    px = float(np.sum(np.diag(mat)))
    p2 = float(np.sum(np.triu(mat, 1)))
    idx_i, idx_j = np.indices((7, 7))
    po25 = float(np.sum(mat[(idx_i + idx_j) > 2]))
    pu25 = float(np.sum(mat[(idx_i + idx_j) <= 2]))
    p_btts = float(np.sum(mat[(idx_i >= 1) & (idx_j >= 1)]))
    return {'1': p1, 'X': px, '2': p2, 'OVER 2.5': po25, 'UNDER 2.5': pu25, 'GOAL (BTTS)': p_btts}

def p_close_pulita(rec, mkt):
    if mkt in ('1', 'X', '2'):
        p = devig_power_singolo([rec.get('AvgCH'), rec.get('AvgCD'), rec.get('AvgCA')])
        idx_map = {'1': 0, 'X': 1, '2': 2}
        val = p.item(idx_map[mkt])
        return float(val) if pd.notna(val) else np.nan
    if mkt in ('OVER 2.5', 'UNDER 2.5'):
        p = devig_power_singolo([rec.get('AvgC_Over25'), rec.get('AvgC_Under25')])
        val = p.item(0) if mkt == 'OVER 2.5' else p.item(1)
        return float(val) if pd.notna(val) else np.nan
    return np.nan


# BLOCCO 3: Calcolo Proxy xG, Rating Elo, Riposo e addestramento modelli Logit-Offset
def costruisci_motore(forza_aggiornamento=False, usa_cache=True):
    oggi_dt = pd.Timestamp.now()
    anno_corr = oggi_dt.year if oggi_dt.month >= 7 else oggi_dt.year - 1
    stagione_corr = f"{str(anno_corr)[-2:]}{str(anno_corr + 1)[-2:]}"
    stagioni_pass = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(2019, anno_corr)]
    file_cache_dinamico = f"cache_prono_mel_fino_{stagioni_pass[-1]}.csv"

    if usa_cache and os.path.exists(file_cache_dinamico) and not forza_aggiornamento:
        df_pass = pd.read_csv(file_cache_dinamico)
        if 'Stagione' not in df_pass.columns:
            df_pass['Stagione'] = 'pass'
    else:
        df_pass, falliti = scarica_stagioni(stagioni_pass)
        if usa_cache and not falliti and not df_pass.empty:
            df_pass.to_csv(file_cache_dinamico, index=False)


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

    db['Proxy_xGH'] = np.clip(0.55 * (db['HST'].fillna(db['FTHG'] * 3.0) * 0.32) + 0.45 * db['FTHG'], 0.1, 3.5)
    db['Proxy_xGA'] = np.clip(0.55 * (db['AST'].fillna(db['FTAG'] * 3.0) * 0.32) + 0.45 * db['FTAG'], 0.1, 3.5)

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


# BLOCCO 4: Funzioni ausiliarie di estrazione segnali per Motore A (Logit) e Motore B (Poisson)
def valuta_match_logit(row, db, elo_ratings, ultima_data, ultima_div, medie_feat, scaler, modelli, mappa_quote, soglia_ev, flat_stake):
    cod = row['Div']
    nome_lega, tier_lega, sport_key = CAMPIONATI[cod]
    casa, trasf, data_str, dt_m = row['HomeTeam'], row['AwayTeam'], str(row['Date']), row['Data_DT']
    _, avg1, _ = mappa_quote['1']
    _, avgX, _ = mappa_quote['X']
    _, avg2, _ = mappa_quote['2']
    _, avgo, _ = mappa_quote['OVER 2.5']
    _, avgu, _ = mappa_quote['UNDER 2.5']

    promozioni = {'I2': 'I1', 'E1': 'E0', 'SP2': 'SP1', 'D2': 'D1', 'F2': 'F1'}
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
        'H_EWM_ResPts': pc['H_Res_Pts'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pc) > 0 else medie_feat['H_EWM_ResPts'],
        'A_EWM_ResPts': pt['A_Res_Pts'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pt) > 0 else medie_feat['A_EWM_ResPts'],
        'H_EWM_SOT_Diff': pc['H_SOT_Diff'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pc) > 0 else medie_feat['H_EWM_SOT_Diff'],
        'A_EWM_SOT_Diff': pt['A_SOT_Diff'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pt) > 0 else medie_feat['A_EWM_SOT_Diff'],
        'H_EWM_GF': pc['FTHG'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pc) > 0 else medie_feat['H_EWM_GF'],
        'A_EWM_GF': pt['FTAG'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pt) > 0 else medie_feat['A_EWM_GF'],
        'Elo_Diff': rh - ra, 'Abs_Elo_Diff': abs(rh - ra),
        'Rest_Diff': rest_h - rest_a, 'Lega_Gol': lega_mg
    }]).fillna(medie_feat)

    eps = 1e-5
    X_s = scaler.transform(row_x)
    p_mkt_1x2 = devig_power_singolo([avg1, avgX, avg2])
    p_raw, adj_logit = {}, {}
    for idx_m, m_nome in enumerate(['1', 'X', '2']):
        b0, b = modelli[m_nome]
        delta_m = float(b0 + (X_s @ b).item(0))
        adj_logit[m_nome] = delta_m
        off = float(logit(np.clip(p_mkt_1x2.item(idx_m), eps, 1 - eps)))
        p_raw[m_nome] = float(expit(off + delta_m))
    s1x2 = p_raw['1'] + p_raw['X'] + p_raw['2']
    p_fin = {'1': p_raw['1']/s1x2, 'X': p_raw['X']/s1x2, '2': p_raw['2']/s1x2}

    if pd.notna(avgo) and pd.notna(avgu) and avgo > 1.0 and avgu > 1.0:
        p_mkt_ou = devig_power_singolo([avgo, avgu])
        for idx_o, m_nome in enumerate(['OVER 2.5', 'UNDER 2.5']):
            b0, b = modelli[m_nome]
            delta_o = float(b0 + (X_s @ b).item(0))
            adj_logit[m_nome] = delta_o
            off = float(logit(np.clip(p_mkt_ou.item(idx_o), eps, 1 - eps)))
            p_raw[m_nome] = float(expit(off + delta_o))
        sou = p_raw['OVER 2.5'] + p_raw['UNDER 2.5']
        p_fin['OVER 2.5'], p_fin['UNDER 2.5'] = p_raw['OVER 2.5']/sou, p_raw['UNDER 2.5']/sou

    res_list = []
    for m_nome, p_ai in p_fin.items():
        qb, qavg, qmax = mappa_quote[m_nome]
        q_rif = qmax if pd.notna(qmax) else qb
        if pd.notna(q_rif) and QUOTA_MIN <= q_rif <= QUOTA_MAX:
            ev = (p_ai * q_rif) - 1.0
            if ev >= soglia_ev:
                res_list.append({
                    'Motore': 'Logit-Offset', 'Data_DT': dt_m, 'Data': data_str, 'Ora_ITA': str(row.get('Ora_ITA', 'N/D')),
                    'Kickoff_ITA': row['Kickoff_ITA'], 'Campionato': nome_lega, 'Tier': tier_lega,
                    'Sport_Key': sport_key, 'Casa': casa, 'Trasferta': trasf,
                    'Partita': f"{casa} - {trasf}", 'Mercato': m_nome,
                    'Quota_B365': qb, 'Quota_Media': round(qavg, 2), 'Quota_MAX': qmax,
                    'Prob_AI_Pct': round(p_ai * 100.0, 2), 'EV_Pct': round(ev * 100.0, 2),
                    'Puntata_Consigliata': flat_stake,
                    'Adj_1': adj_logit.get('1', 0.0), 'Adj_X': adj_logit.get('X', 0.0), 'Adj_2': adj_logit.get('2', 0.0),
                    'Adj_O25': adj_logit.get('OVER 2.5', 0.0), 'Adj_U25': adj_logit.get('UNDER 2.5', 0.0)
                })
    return res_list

def valuta_match_poisson(row, db, stagione_corr, mappa_quote):
    cod = row['Div']
    if cod not in LEGHE_POISSON:
        return []
    nome_lega, tier_lega, sport_key = CAMPIONATI[cod]
    casa, trasf, data_str, dt_m = row['HomeTeam'], row['AwayTeam'], str(row['Date']), row['Data_DT']
    st_val = db[db['Data_DT'] < dt_m]
    st_stag = st_val[(st_val['Div'] == cod) & (st_val['Stagione'] == stagione_corr)]
    n_g_casa = len(st_stag[(st_stag['HomeTeam'] == casa) | (st_stag['AwayTeam'] == casa)])
    n_g_trasf = len(st_stag[(st_stag['HomeTeam'] == trasf) | (st_stag['AwayTeam'] == trasf)])
    if n_g_casa < 4 or n_g_trasf < 4:
        return []

    lega_sub = st_val[st_val['Div'] == cod].tail(380)
    mu_h = float(lega_sub['Proxy_xGH'].mean())
    mu_a = float(lega_sub['Proxy_xGA'].mean())
    pc_p = st_val[st_val['HomeTeam'] == casa].tail(15)
    pt_p = st_val[st_val['AwayTeam'] == trasf].tail(15)
    if len(pc_p) < 2 or len(pt_p) < 2 or mu_h <= 0 or mu_a <= 0:
        return []

    as_h = (0.70 * float(pc_p['Proxy_xGH'].tail(5).mean()) + 0.30 * float(pc_p['Proxy_xGH'].mean())) / mu_h
    dw_h = (0.70 * float(pc_p['Proxy_xGA'].tail(5).mean()) + 0.30 * float(pc_p['Proxy_xGA'].mean())) / mu_a
    as_a = (0.70 * float(pt_p['Proxy_xGA'].tail(5).mean()) + 0.30 * float(pt_p['Proxy_xGA'].mean())) / mu_a
    dw_a = (0.70 * float(pt_p['Proxy_xGH'].tail(5).mean()) + 0.30 * float(pt_p['Proxy_xGH'].mean())) / mu_h

    lam_h = float(np.clip(as_h * dw_a * mu_h, 0.25, 4.2))
    lam_a = float(np.clip(as_a * dw_h * mu_a, 0.25, 4.2))
    p_pois = calcola_poisson_dixon_coles(lam_h, lam_a)

    res_list = []
    for m_nome in ['1', 'X', '2', 'OVER 2.5', 'UNDER 2.5']:
        qb, qavg, qmax = mappa_quote[m_nome]
        q_rif = qmax if pd.notna(qmax) else qb
        p_ai = p_pois[m_nome]
        if pd.notna(q_rif) and 1.65 <= q_rif <= 2.40:
            ev = (p_ai * q_rif) - 1.0
            if 0.09 <= ev <= 0.22:
                res_list.append({
                    'Motore': 'Poisson-xG', 'Data_DT': dt_m, 'Data': data_str, 'Ora_ITA': str(row.get('Ora_ITA', 'N/D')),
                    'Kickoff_ITA': row['Kickoff_ITA'], 'Campionato': nome_lega, 'Tier': tier_lega,
                    'Sport_Key': sport_key, 'Casa': casa, 'Trasferta': trasf,
                    'Partita': f"{casa} - {trasf}", 'Mercato': m_nome,
                    'Quota_B365': qb, 'Quota_Media': round(qavg, 2), 'Quota_MAX': qmax,
                    'Prob_AI_Pct': round(p_ai * 100.0, 2), 'EV_Pct': round(ev * 100.0, 2),
                    'Puntata_Consigliata': 5.0,
                    'Adj_1': 0.0, 'Adj_X': 0.0, 'Adj_2': 0.0,
                    'Adj_O25': 0.0, 'Adj_U25': 0.0
                })
    return res_list


# BLOCCO 5: Lettura palinsesto futuro, ordinamento cronologico e cap giornaliero
def analizza_palinsesto_futuro(db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli, soglia_ev, flat_stake, budget_tot, solo_top5=True, modalita_motore="Entrambi (Logit + Poisson)"):
    
    url_fix = "https://www.football-data.co.uk/fixtures.csv"
    txt_fix = scarica_csv_veloce(url_fix, tentativi=2)
    if txt_fix is None:
        return pd.DataFrame(), "Impossibile scaricare fixtures.csv al momento."
    try:
        df_fix = pd.read_csv(io.StringIO(txt_fix), on_bad_lines='skip', encoding='utf-8-sig', engine='python')
        df_fix.columns = df_fix.columns.str.replace('ï»¿', '')
        if 'Div' not in df_fix.columns:
            raise KeyError("Colonna Div non trovata")
        df_fix = rinomina_colonne_csv(df_fix)
    except Exception as e:
        return pd.DataFrame(), f"Errore lettura palinsesto: {e}"

    df_fix = df_fix[df_fix['Div'].isin(list(CAMPIONATI.keys()))].copy()
    for c_b, c_avg in [('B365H', 'AvgH'), ('B365D', 'AvgD'), ('B365A', 'AvgA')]:
        if c_avg in df_fix.columns:
            df_fix[c_b] = df_fix[c_b].fillna(df_fix[c_avg])
    df_fix = df_fix.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']).copy()
    df_fix['Data_DT'] = pd.to_datetime(df_fix['Date'], dayfirst=True, format='mixed', errors='coerce')

    if 'Time' in df_fix.columns:
        dt_uk = pd.to_datetime(df_fix['Date'].astype(str) + ' ' + df_fix['Time'].fillna('15:00').astype(str), dayfirst=True, format='mixed', errors='coerce')
        df_fix['Kickoff_ITA'] = dt_uk.dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('Europe/Rome')
        df_fix['Ora_ITA'] = df_fix['Kickoff_ITA'].dt.strftime('%H:%M').fillna(df_fix['Time'].fillna('N/D'))
    else:
        df_fix['Kickoff_ITA'] = pd.NaT
        df_fix['Ora_ITA'] = 'N/D'

    oggi = pd.Timestamp.now().normalize()
    df_fix = df_fix[df_fix['Data_DT'] >= oggi].copy()

    if solo_top5:
        df_fix = df_fix[df_fix['Div'].isin(['I1', 'E0', 'SP1', 'D1', 'F1'])].copy()
    if df_fix.empty:
        return pd.DataFrame(), "Nessuna partita futura in programma per i campionati selezionati."

    oggi_dt = pd.Timestamp.now()
    anno_corr = oggi_dt.year if oggi_dt.month >= 7 else oggi_dt.year - 1
    stagione_corr = f"{str(anno_corr)[-2:]}{str(anno_corr + 1)[-2:]}"

    usa_logit = "Logit" in modalita_motore or "Entrambi" in modalita_motore
    usa_poisson = "Poisson" in modalita_motore or "Entrambi" in modalita_motore
    segnali = []

    for _, row in df_fix.iterrows():
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

        mappa_quote = {
            '1': (q1, avg1, max1), 'X': (qX, avgX, maxX), '2': (q2, avg2, max2),
            'OVER 2.5': (qo, avgo, maxo), 'UNDER 2.5': (qu, avgu, maxu)
        }
        if usa_logit:
            segnali.extend(valuta_match_logit(row, db, elo_ratings, ultima_data, ultima_div, medie_feat, scaler, modelli, mappa_quote, soglia_ev, flat_stake))
        if usa_poisson:
            segnali.extend(valuta_match_poisson(row, db, stagione_corr, mappa_quote))

    if not segnali:
        return pd.DataFrame(), "Nessun segnale supera i filtri di sicurezza dei motori selezionati."

    df_seg = pd.DataFrame(segnali).sort_values(by=['Data_DT', 'Ora_ITA', 'EV_Pct'], ascending=[True, True, False]).reset_index(drop=True)
    cap_euro = budget_tot * MAX_ESPOSIZIONE_GIORNO_PCT
    for dt_v, grp in df_seg[df_seg['Motore'] == 'Logit-Offset'].groupby('Data'):
        tot_d = grp['Puntata_Consigliata'].sum()
        if tot_d > cap_euro and tot_d > 0:
            df_seg.loc[grp.index, 'Puntata_Consigliata'] = (df_seg.loc[grp.index, 'Puntata_Consigliata'] * (cap_euro / tot_d)).round(2)

    return df_seg, "OK"


