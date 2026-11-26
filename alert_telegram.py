# BOT ALERT TELEGRAM: Motore condiviso + Esclusione automatica partite già giocate o scartate
import os
import io
import requests
import pandas as pd
from motore import costruisci_motore, analizza_palinsesto_futuro, norm_data

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GSHEET_ID = os.environ.get("GSHEET_ID", "")

def carica_gia_gestite():
    gia_gestite = set()
    if not GSHEET_ID:
        return gia_gestite
    for foglio in ["Tracker", "Scarti"]:
        try:
            url_csv = "https://docs." + "google.com" + f"/spreadsheets/d/{GSHEET_ID}/gviz/tq?tqx=out:csv&sheet={foglio}"
            df_f = pd.read_csv(url_csv)
            if not df_f.empty and 'Partita' in df_f.columns and 'Mercato' in df_f.columns:
                for _, r_f in df_f.iterrows():
                    dm_f = norm_data(r_f['Data_Match']) if 'Data_Match' in df_f.columns and pd.notna(r_f.get('Data_Match')) else ''

                    gia_gestite.add((str(r_f['Partita']).strip(), str(r_f['Mercato']).strip(), dm_f))
        except Exception:
            pass
    return gia_gestite

def invia_tg(testo):
    if not TOKEN or not CHAT_ID:
        return
    url_tg = "https://api." + "telegram.org" + f"/bot{TOKEN}/sendMessage"
    for i in range(0, len(testo), 3800):
        r = requests.post(url_tg, json={"chat_id": CHAT_ID, "text": testo[i:i+3800]}, timeout=15)
        r.raise_for_status()


def esegui_controllo():
    oggi = pd.Timestamp.now(tz="Europe/Rome")
    giorno = oggi.weekday()  # 0=Lun, 3=Gio, 4=Ven, 5=Sab, 6=Dom
    ora_corr = oggi.hour

    if giorno == 0:
        if ora_corr == 10:
            invia_tg("📅 LUNEDÌ — PRONO MEL v1\nRicordati di aprire l'app e premere 'Chiudi Partite Concluse in Automatico' nella Scheda 2 per aggiornare CLV, Confronto Strategie e Semaforo!")
        return

    if giorno not in (3, 4, 5, 6):
        return
    if giorno == 3 and ora_corr != 18:
        return
    if giorno == 4 and ora_corr not in (11, 18):
        return
        
    if giorno in (5, 6):
        if not (10 <= ora_corr <= 21):
            return
        try:
            url_fix = "https://www." + "football-data.co.uk" + "/fixtures.csv"
            r_fix = requests.get(url_fix, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            r_fix.raise_for_status()
            df_chk = pd.read_csv(io.StringIO(r_fix.text), on_bad_lines='skip', encoding='utf-8-sig', engine='python')
            df_chk.columns = df_chk.columns.str.replace('ï»¿', '').str.strip()
            df_chk = df_chk[df_chk['Div'].isin(['I1', 'E0', 'SP1', 'D1', 'F1'])].copy()
            dt_uk = pd.to_datetime(df_chk['Date'].astype(str) + ' ' + df_chk['Time'].fillna('15:00').astype(str), dayfirst=True, format='mixed', errors='coerce')
            ko_ita = dt_uk.dt.tz_localize('Europe/London', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('Europe/Rome')
            min_manc = (ko_ita - oggi).dt.total_seconds() / 60.0
            if not ((min_manc >= 20) & (min_manc <= 79)).any():
                return
        except Exception as e:
            print(f"Pre-check fixtures saltato: {e}")

    db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli = costruisci_motore(usa_cache=False)
    df_seg, msg = analizza_palinsesto_futuro(
        db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli,
        soglia_ev=0.03, flat_stake=10.0, budget_tot=1000.0, solo_top5=True,
        modalita_motore="Entrambi (Logit + Poisson)"
    )

    if not df_seg.empty:
        gia_gestite = carica_gia_gestite()
        if gia_gestite:
            df_seg = df_seg[~df_seg.apply(lambda r: (
                (str(r['Partita']).strip(), str(r['Mercato']).strip(), norm_data(r['Data'])) in gia_gestite or
                (str(r['Partita']).strip(), str(r['Mercato']).strip(), '') in gia_gestite
            ), axis=1)].reset_index(drop=True)



    if df_seg.empty:
        if giorno in (3, 4):
            invia_tg("⚽ PRONO MEL v1: Nessun nuovo segnale disponibile (o tutte le partite estratte sono già state giocate/scartate).")
        return

    trovate = []
    imminenti = []
    for _, r in df_seg.iterrows():
        
        riga_msg = f"[{r['Motore']}] 📅 {r['Data']} ({r['Ora_ITA']}) | [{r['Campionato']}]\n⚽ {r['Partita']} -> {r['Mercato']} @ {r['Quota_MAX']} (EV: +{r['EV_Pct']}% | su Media: {r.get('EV_su_Media_Pct', 'N/D')}%)"

        trovate.append(riga_msg)
        ko_ita = r['Kickoff_ITA']
        if pd.notna(ko_ita):
            minuti_mancanti = (ko_ita - oggi).total_seconds() / 60.0
            if 20 <= minuti_mancanti <= 79:
                imminenti.append(riga_msg)

    if giorno in (5, 6):
        if imminenti:
            msg_pre = "\n\n".join(imminenti)
            invia_tg(f"⏱️ ALERT PRE-PARTITA (CHECK FORMAZIONI T-45 min)\n\n{msg_pre}\n\n👉 Controlla formazioni su Diretta.it e verifica in Prono Mel v1!")
        return

    msg_wk = "\n\n".join(trovate)
    invia_tg(f"⚽ PRONO MEL v1 — SEGNALI WEEKEND\n\n{msg_wk}\n\n👉 Apri Prono Mel v1 per Line Shopping e verifica!")

if __name__ == "__main__":
    esegui_controllo()
