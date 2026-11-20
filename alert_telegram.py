import os
import io
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.football-data.co.uk"
}

CAMPIONATI_TOP5 = {
    'I1': 'Serie A', 'E0': 'Premier League', 'SP1': 'La Liga',
    'D1': 'Bundesliga', 'F1': 'Ligue 1'
}

def invia_tg(testo):
    if TOKEN and CHAT_ID:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": testo, "parse_mode": "Markdown"},
            timeout=15
        )

def devig_power_col(df, cols):
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
def esegui_controllo():
    oggi = pd.Timestamp.now(tz="Europe/Rome")
    giorno = oggi.weekday()  # 0=Lun, 3=Gio, 4=Ven, 5=Sab, 6=Dom
    ora_corr = oggi.hour

    if giorno == 0:
        if ora_corr == 10:
            invia_tg("📅 *LUNEDÌ — PRONO MEL v1*\nRicordati di aprire l'app e premere *🔄 Chiudi Partite Concluse in Automatico* nella Scheda 2 per aggiornare CLV e Semaforo!")
        return

    if giorno not in (3, 4, 5, 6):
        return
    if giorno == 3 and ora_corr != 18:
        return
    if giorno == 4 and ora_corr not in (11, 18):
        return

    anno = oggi.year if oggi.month >= 7 else oggi.year - 1

    stagioni = [f"{str(y)[-2:]}{str(y + 1)[-2:]}" for y in range(2020, anno + 1)]
    fogli = []
    for cod in CAMPIONATI_TOP5.keys():
        for s in stagioni:
            try:
                url_csv = "https://www.football-data.co.uk" + f"/mmz4281/{s}/{cod}.csv"

                r = requests.get(url_csv, headers=HEADERS, timeout=15)
                if r.status_code == 200 and "<html" not in r.text.lower():
                    df = pd.read_csv(io.StringIO(r.text), on_bad_lines='skip', encoding='utf-8-sig', engine='python')
                    df.columns = df.columns.str.replace('ï»¿', '')
                    if 'Div' not in df.columns:
                        raise KeyError(f"Colonna 'Div' mancante. Colonne: {list(df.columns)}")          
                    if all(c in df.columns for c in ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']):
                        for c in ['HST', 'AST', 'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA']:
                            if c not in df.columns:
                                df[c] = np.nan
                        df['Div'] = cod
                        fogli.append(df[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HST', 'AST', 'B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA']])
            except Exception as e:
                print(f"Errore nel file {cod} stagione {s}: {e}")            
            

    if not fogli:
        invia_tg("⚠️ *PRONO MEL v1*: Impossibile scaricare i CSV da football-data in questo momento.")
        return

    db = pd.concat(fogli, ignore_index=True).dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A'])
    db['Data_DT'] = pd.to_datetime(db['Date'], dayfirst=True, format='mixed', errors='coerce')
    db = db.dropna(subset=['Data_DT']).sort_values(['Data_DT', 'Div', 'HomeTeam']).reset_index(drop=True)
    for c_alt, c_b in [('AvgH', 'B365H'), ('AvgD', 'B365D'), ('AvgA', 'B365A'), ('MaxH', 'B365H'), ('MaxD', 'B365D'), ('MaxA', 'B365A')]:
        db[c_alt] = db[c_alt].fillna(db[c_b])

    p_mkt = devig_power_col(db, ['AvgH', 'AvgD', 'AvgA'])
    db['Mkt_P1'], db['Mkt_PX'], db['Mkt_P2'] = p_mkt[:, 0], p_mkt[:, 1], p_mkt[:, 2]
    db['H_Res_Pts'] = np.where(db['FTHG'] > db['FTAG'], 3, np.where(db['FTHG'] == db['FTAG'], 1, 0)) - (3 * db['Mkt_P1'] + db['Mkt_PX'])
    db['A_Res_Pts'] = np.where(db['FTAG'] > db['FTHG'], 3, np.where(db['FTAG'] == db['FTHG'], 1, 0)) - (3 * db['Mkt_P2'] + db['Mkt_PX'])
    db['H_SOT_Diff'] = db['HST'] - db['AST']
    db['A_SOT_Diff'] = db['AST'] - db['HST']
    db['H_EWM_ResPts'] = db.groupby('HomeTeam')['H_Res_Pts'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['A_EWM_ResPts'] = db.groupby('AwayTeam')['A_Res_Pts'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['H_EWM_SOT'] = db.groupby('HomeTeam')['H_SOT_Diff'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())
    db['A_EWM_SOT'] = db.groupby('AwayTeam')['A_SOT_Diff'].transform(lambda x: x.shift(1).ewm(span=10, min_periods=1).mean())

    feat = ['H_EWM_ResPts', 'A_EWM_ResPts', 'H_EWM_SOT', 'A_EWM_SOT']
    medie = db[feat].mean()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(db[feat].fillna(medie))
    eps = 1e-5
    mod = {}
    for m, (col_y, col_m) in {'1': ((db['FTHG'] > db['FTAG']).astype(int), 'Mkt_P1'), 'X': ((db['FTHG'] == db['FTAG']).astype(int), 'Mkt_PX'), '2': ((db['FTHG'] < db['FTAG']).astype(int), 'Mkt_P2')}.items():
        v = db[col_m].notna().values
        yv, offv, Xv = col_y.values[v], logit(np.clip(db.loc[v, col_m].values, eps, 1 - eps)), Xs[v]
        def lg(par):
            p = expit(offv + par.item(0) + Xv @ par[1:])
            ll = -np.mean(yv * np.log(np.clip(p, eps, 1 - eps)) + (1 - yv) * np.log(np.clip(1 - p, eps, 1 - eps))) + 0.5 * (400.0 / len(yv)) * np.sum(par[1:] ** 2)
            d = (p - yv) / len(yv)
            return ll, np.concatenate([[np.sum(d)], Xv.T @ d + (400.0 / len(yv)) * par[1:]])
        res = minimize(lg, np.zeros(len(feat) + 1), jac=True, method='BFGS')
        mod[m] = (float(res.x.item(0)), res.x[1:])

    url_fix = "https://www.football-data.co.uk/fixtures.csv"
    r_fix = requests.get(url_fix, headers=HEADERS, timeout=15)
    try:
        df_fix = pd.read_csv(io.StringIO(r_fix.text), on_bad_lines='skip', encoding='utf-8-sig', engine='python')
        df_fix.columns = df_fix.columns.str.replace('ï»¿', '')
        if 'Div' not in df_fix.columns:
            raise KeyError("Colonna Div non trovata")
        df_fix = df_fix[df_fix['Div'].isin(list(CAMPIONATI_TOP5.keys()))].dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A'])
        df_fix['Data_DT'] = pd.to_datetime(df_fix['Date'], dayfirst=True, format='mixed', errors='coerce')
        if 'Time' in df_fix.columns:
            dt_uk = pd.to_datetime(df_fix['Date'].astype(str) + ' ' + df_fix['Time'].fillna('15:00').astype(str), dayfirst=True, format='mixed', errors='coerce')
            df_fix['Kickoff_ITA'] = dt_uk.dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('Europe/Rome')
        else:
            df_fix['Kickoff_ITA'] = pd.NaT
        df_fix = df_fix[df_fix['Data_DT'] >= pd.Timestamp.now().normalize()]
    except Exception as e:
        invia_tg(f"⚠️ *ERRORE LETTURA PALINSESTO*: {e}")
        return

    trovate = []
    imminenti = []
    for _, row in df_fix.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        avg_q = [float(row['AvgH']) if pd.notna(row.get('AvgH')) else float(row['B365H']),
                 float(row['AvgD']) if pd.notna(row.get('AvgD')) else float(row['B365D']),
                 float(row['AvgA']) if pd.notna(row.get('AvgA')) else float(row['B365A'])]
        max_q = {'1': float(row['MaxH']) if pd.notna(row.get('MaxH')) else float(row['B365H']),
                 'X': float(row['MaxD']) if pd.notna(row.get('MaxD')) else float(row['B365D']),
                 '2': float(row['MaxA']) if pd.notna(row.get('MaxA')) else float(row['B365A'])}
        p_open = devig_power_col(pd.DataFrame([avg_q], columns=['a', 'b', 'c']), ['a', 'b', 'c']).flatten()
        pc, pt = db[db['HomeTeam'] == ht], db[db['AwayTeam'] == at]
        rx = pd.DataFrame([{
            'H_EWM_ResPts': pc['H_Res_Pts'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pc) > 0 else medie['H_EWM_ResPts'],
            'A_EWM_ResPts': pt['A_Res_Pts'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pt) > 0 else medie['A_EWM_ResPts'],
            'H_EWM_SOT': pc['H_SOT_Diff'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pc) > 0 else medie['H_EWM_SOT'],
            'A_EWM_SOT': pt['A_SOT_Diff'].ewm(span=10, min_periods=1).mean().tolist().pop(-1) if len(pt) > 0 else medie['A_EWM_SOT']
        }]).fillna(medie)
        x_s = scaler.transform(rx)
        p_raw = {}
        for i, m in enumerate(['1', 'X', '2']):
            b0_m, b_m = mod[m]
            off_m = float(logit(np.clip(p_open.item(i), eps, 1 - eps)))
            p_raw[m] = float(expit(off_m + b0_m + (x_s @ b_m).item(0)))
        tot_p = sum(p_raw.values())
        ko_ita = row['Kickoff_ITA']
        ora_str = ko_ita.strftime('%H:%M') if pd.notna(ko_ita) else "N/D"

        for m in ['1', 'X', '2']:
            p_f = p_raw[m] / tot_p
            q_m = max_q[m]
            ev = (p_f * q_m) - 1.0
            if 1.35 <= q_m <= 4.00 and ev >= 0.03:
                riga_msg = f"📅 {row['Date']} ({ora_str}) | *{CAMPIONATI_TOP5[row['Div']]}*\n🏟️ {ht} - {at} -> *{m}* @ *{q_m}* (EV: *+{ev*100:.1f}%*)"
                trovate.append(riga_msg)
                if pd.notna(ko_ita):
                    minuti_mancanti = (ko_ita - oggi).total_seconds() / 60.0
                    if 15 <= minuti_mancanti <= 105:
                        imminenti.append(riga_msg)

    if giorno in (5, 6):
        if imminenti:
            msg_pre = "\n\n".join(imminenti)
            invia_tg(f"⏱️ *ALERT PRE-PARTITA (CHECK FORMAZIONI T-45 min)* ⏱️\n\n{msg_pre}\n\n👉 Controlla formazioni su *Diretta.it* e registra in *Prono Mel v1*!")
        return

    if trovate:
        msg_wk = "\n\n".join(trovate)
        invia_tg(f"⚽ *PRONO MEL v1 — SEGNALI WEEKEND* ⚽\n\n{msg_wk}\n\n👉 Apri *Prono Mel v1* per Line Shopping e verifica!")
    else:
        invia_tg("⚽ *PRONO MEL v1*: Nessuna partita nei Top 5 supera EV +3.0% oggi.")

if __name__ == "__main__":
    esegui_controllo()

    
