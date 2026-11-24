# BOT ALERT TELEGRAM: Usa il motore condiviso (Logit + Poisson) con filtro orario anti-doppioni
import os
import requests
import pandas as pd
from motore import costruisci_motore, analizza_palinsesto_futuro

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def invia_tg(testo):
    if not TOKEN or not CHAT_ID:
        return
    url_tg = "https://api." + "telegram.org" + f"/bot{TOKEN}/sendMessage"
    r = requests.post(url_tg, json={"chat_id": CHAT_ID, "text": testo}, timeout=15)
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

    db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli = costruisci_motore(usa_cache=False)
    df_seg, msg = analizza_palinsesto_futuro(
        db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli,
        soglia_ev=0.03, flat_stake=10.0, budget_tot=1000.0, solo_top5=True,
        modalita_motore="Entrambi (Logit + Poisson)"
    )

    if df_seg.empty:
        if giorno in (3, 4):
            invia_tg(f"⚽ PRONO MEL v1: {msg}")
        return

    trovate = []
    imminenti = []
    for _, r in df_seg.iterrows():
        riga_msg = f"[{r['Motore']}] 📅 {r['Data']} ({r['Ora_ITA']}) | {r['Campionato']}\n🏟️ {r['Partita']} -> {r['Mercato']} @ {r['Quota_MAX']} (EV: +{r['EV_Pct']}%)"
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
