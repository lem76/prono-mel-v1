# BLOCCO 1: Importazioni, controllo accesso password e comparatore quote live (The Odds API)
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from scipy.special import logit, expit
from motore import (
    QUOTA_MIN, QUOTA_MAX, MAX_ESPOSIZIONE_GIORNO_PCT,
    CAMPIONATI, devig_power_singolo, costruisci_motore,
    analizza_palinsesto_futuro, p_close_pulita
)

st.set_page_config(page_title="Prono Mel v1 - Calcio AI & Tracker", page_icon="⚽", layout="wide")
if "autenticato" not in st.session_state:
    st.session_state["autenticato"] = False

if not st.session_state["autenticato"]:
    st.title("🔒 Accesso Riservato — Prono Mel v1")
    pwd_inserita = st.text_input("Inserisci la password personale:", type="password")
    if st.button("Entra", type="primary"):
        if pwd_inserita == st.secrets.get("app_password", ""):
            st.session_state["autenticato"] = True
            st.rerun()
        else:
            st.error("Password errata. Riprova.")
    st.stop()

def cerca_migliore_quota_live(api_key, sport_key, casa, trasferta, riga_match):
    if not api_key:
        return None, None, "Inserisci una API Key gratuita di The Odds API nella barra laterale."
    mercato = riga_match['Mercato']
    motore_nome = riga_match.get('Motore', 'Logit-Offset')
    mkt_api = "totals" if "2.5" in mercato else "h2h"
    url = f"https://the-odds-api.com{sport_key}/odds/?apiKey={api_key}&regions=eu,uk&markets={mkt_api}&oddsFormat=decimal"

    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None, None, f"Errore API ({r.status_code}): controlla la chiave."
        dati = r.json()
        casa_low, trasf_low = casa.lower()[:5], trasferta.lower()[:5]
        eps = 1e-5
        for evento in dati:
            h_ev = evento.get('home_team', '').lower()
            a_ev = evento.get('away_team', '').lower()
            if casa_low in h_ev or trasf_low in a_ev:
                offerte = []
                tutti_esiti = {'1': [], 'X': [], '2': [], 'OVER 2.5': [], 'UNDER 2.5': []}
                for bk in evento.get('bookmakers', []):
                    nome_bk = bk.get('title', 'Bookmaker')
                    for m in bk.get('markets', []):
                        for out in m.get('outcomes', []):
                            nome_out = out.get('name', '')
                            prezzo = float(out.get('price', 0.0))
                            punto = out.get('point', None)
                            if prezzo <= 1.0:
                                continue
                            esito_cod = None
                            if nome_out == evento.get('home_team'):
                                esito_cod = '1'
                            elif nome_out == evento.get('away_team'):
                                esito_cod = '2'
                            elif nome_out.lower() == 'draw':
                                esito_cod = 'X'
                            elif nome_out.lower() == 'over' and punto == 2.5:
                                esito_cod = 'OVER 2.5'
                            elif nome_out.lower() == 'under' and punto == 2.5:
                                esito_cod = 'UNDER 2.5'
                            if esito_cod:
                                tutti_esiti[esito_cod].append(prezzo)
                                if esito_cod == mercato:
                                    offerte.append({'Bookmaker': nome_bk, 'Quota Live': prezzo})

                prob_live_ai = None
                if motore_nome == 'Logit-Offset':
                    if mercato in ('1', 'X', '2') and tutti_esiti['1'] and tutti_esiti['X'] and tutti_esiti['2']:
                        med_q = [float(np.median(tutti_esiti['1'])), float(np.median(tutti_esiti['X'])), float(np.median(tutti_esiti['2']))]
                        p_mkt_live = devig_power_singolo(med_q)
                        adjs = [float(riga_match.get('Adj_1', 0.0)), float(riga_match.get('Adj_X', 0.0)), float(riga_match.get('Adj_2', 0.0))]
                        p_raw_l = [float(expit(float(logit(np.clip(p_mkt_live.item(i), eps, 1 - eps))) + adjs[i])) for i in range(3)]
                        s_l = sum(p_raw_l)
                        idx_m = {'1': 0, 'X': 1, '2': 2}[mercato]
                        prob_live_ai = round((p_raw_l[idx_m] / s_l) * 100.0, 2)
                    elif mercato in ('OVER 2.5', 'UNDER 2.5') and tutti_esiti['OVER 2.5'] and tutti_esiti['UNDER 2.5']:
                        med_q = [float(np.median(tutti_esiti['OVER 2.5'])), float(np.median(tutti_esiti['UNDER 2.5']))]
                        p_mkt_live = devig_power_singolo(med_q)
                        adjs = [float(riga_match.get('Adj_O25', 0.0)), float(riga_match.get('Adj_U25', 0.0))]
                        p_raw_l = [float(expit(float(logit(np.clip(p_mkt_live.item(i), eps, 1 - eps))) + adjs[i])) for i in range(2)]
                        s_l = sum(p_raw_l)
                        idx_o = {'OVER 2.5': 0, 'UNDER 2.5': 1}[mercato]
                        prob_live_ai = round((p_raw_l[idx_o] / s_l) * 100.0, 2)

                if offerte:
                    df_off = pd.DataFrame(offerte).sort_values(by='Quota Live', ascending=False).reset_index(drop=True)
                    return df_off, prob_live_ai, "OK"
        return None, None, "Partita non ancora quotata su The Odds API."
    except Exception as e:
        return None, None, f"Errore connessione: {e}"


# BLOCCO 2: Gestione Google Sheets (Tracker, Scarti, Confronto_Strategie, Curva_Equity) e chiusura automatica
def get_gsheets_conn():
    try:
        return st.connection("gsheets", type=GSheetsConnection)
    except Exception:
        return None

def carica_tracker():
    colonne = [
        'ID', 'Data_Log', 'Data_Match', 'Campionato', 'Tier', 'Partita', 'Mercato',
        'Variante_Modello', 'Bookmaker', 'Quota_Presa', 'Quota_Chiusura',
        'Prob_AI_Pct', 'Malus_Assenze_Pct', 'Puntata_Euro', 'Esito', 'Profitto_Euro', 'CLV_Netto_Pct'
    ]
    conn = get_gsheets_conn()
    if conn is not None:
        try:
            df = conn.read(worksheet="Tracker", ttl=0)
            if df is not None and not df.empty:
                df = df.dropna(how='all').reset_index(drop=True)
                if 'Data_Log' not in df.columns:
                    df.insert(1, 'Data_Log', '')
                return df
        except Exception:
            pass
    if os.path.exists("prono_mel_tracker.csv"):
        df = pd.read_csv("prono_mel_tracker.csv")
        if 'Data_Log' not in df.columns:
            df.insert(1, 'Data_Log', '')
        return df
    return pd.DataFrame(columns=colonne)

def genera_tabelle_comparative(df_tr, budget_base=1000.0):
    chiuse = df_tr[df_tr['Esito'].isin(['Vinta', 'Persa'])].copy()
    if chiuse.empty:
        return pd.DataFrame(), pd.DataFrame()

    righe_comp = []
    for var_nome, grp in chiuse.groupby('Variante_Modello'):
        n_g = len(grp)
        n_v = int((grp['Esito'] == 'Vinta').sum())
        wr = round((n_v / n_g) * 100.0, 1) if n_g > 0 else 0.0
        vol = float(pd.to_numeric(grp['Puntata_Euro'], errors='coerce').sum())
        prof = float(pd.to_numeric(grp['Profitto_Euro'], errors='coerce').sum())
        roi_v = round((prof / vol) * 100.0, 2) if vol > 0 else 0.0
        clv_g = pd.to_numeric(grp['CLV_Netto_Pct'], errors='coerce').dropna()
        clv_v = round(float(clv_g.mean()), 2) if not clv_g.empty else np.nan
        righe_comp.append({
            'Strategia_Variante': var_nome, 'Chiuse': n_g, 'Win_Rate_Pct': wr,
            'Volume_Euro': round(vol, 2), 'Profitto_Euro': round(prof, 2),
            'ROI_Pct': roi_v, 'CLV_Netto_Pct': clv_v
        })
    df_comp = pd.DataFrame(righe_comp).sort_values(by='CLV_Netto_Pct', ascending=False).reset_index(drop=True)

    chiuse['Prof_Num'] = pd.to_numeric(chiuse['Profitto_Euro'], errors='coerce').fillna(0.0)
    chiuse['Is_Motore_B'] = chiuse['Variante_Modello'].astype(str).str.contains('Motore B', na=False)
    prof_a = np.where(~chiuse['Is_Motore_B'], chiuse['Prof_Num'], 0.0).cumsum()
    prof_b = np.where(chiuse['Is_Motore_B'], chiuse['Prof_Num'], 0.0).cumsum()
    prof_tot = chiuse['Prof_Num'].cumsum().values

    df_eq = pd.DataFrame({
        'Numero_Giocata': range(1, len(chiuse) + 1),
        'Data_Match': chiuse['Data_Match'].values,
        'Capitale_Motore_A_Euro': np.round(budget_base + prof_a, 2),
        'Capitale_Motore_B_Euro': np.round(budget_base + prof_b, 2),
        'Capitale_Totale_Euro': np.round(budget_base + prof_tot, 2)
    })
    return df_comp, df_eq

def salva_tracker(df_tr, budget_base=1000.0):
    df_tr.to_csv("prono_mel_tracker.csv", index=False)
    conn = get_gsheets_conn()
    if conn is not None:
        try:
            conn.update(worksheet="Tracker", data=df_tr)
            df_comp, df_eq = genera_tabelle_comparative(df_tr, budget_base=budget_base)
            if not df_comp.empty:
                conn.update(worksheet="Confronto_Strategie", data=df_comp)
            if not df_eq.empty:
                conn.update(worksheet="Curva_Equity", data=df_eq)
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

@st.cache_resource(show_spinner="Caricamento storico e addestramento motori Prono Mel v1...")
def prepara_motore_predittivo(forza_aggiornamento=False):
    return costruisci_motore(forza_aggiornamento=forza_aggiornamento, usa_cache=True)

def invia_alert_telegram(df_seg):
    token = st.secrets.get("telegram_token", "")
    chat_id = str(st.secrets.get("telegram_chat_id", ""))
    if not token or not chat_id:
        return False, "Configura telegram_token e telegram_chat_id nei Secrets di Streamlit."
    righe = ["⚽ PRONO MEL v1 — SEGNALI RILEVATI\n"]
    for _, r in df_seg.iterrows():
        righe.append(
            f"[{r.get('Motore', 'Logit')}] 📅 {r['Data']} ({r.get('Ora_ITA', 'N/D')}) | {r['Campionato']}\n"
            f"🏟️ {r['Partita']} -> Mercato: {r['Mercato']}\n"
            f"💰 Quota Max: {r['Quota_MAX']} | EV: +{r['EV_Pct']}% | Puntata: {r['Puntata_Consigliata']} €\n"
        )
    testo = "\n".join(righe)
    url_tg = "https://api." + "telegram.org" + f"/bot{token}/sendMessage"
    try:
        resp = requests.post(url_tg, json={"chat_id": chat_id, "text": testo}, timeout=10)
        if resp.status_code == 200:
            return True, "Alert inviato con successo su Telegram!"
        return False, f"Errore Telegram ({resp.status_code}): avvia prima la chat con il tuo bot."
    except Exception as e:
        return False, f"Errore invio Telegram: {e}"

def chiudi_partite_automatico(df_tr, db_storico):
    in_corso = df_tr[df_tr['Esito'] == 'In Corso']
    if in_corso.empty:
        return df_tr, 0
    chiuse_count = 0
    col_q_close = {'1': 'AvgCH', 'X': 'AvgCD', '2': 'AvgCA', 'OVER 2.5': 'AvgC_Over25', 'UNDER 2.5': 'AvgC_Under25'}
    for idx in in_corso.index:
        partita_str = str(df_tr.loc[idx, 'Partita'])
        if " - " not in partita_str:
            continue
        casa, trasf = partita_str.split(" - ", 1)
        data_dt = pd.to_datetime(df_tr.loc[idx, 'Data_Match'], dayfirst=True, format='mixed', errors='coerce')
        if pd.isna(data_dt):
            continue
        match_db = db_storico[
            (db_storico['HomeTeam'] == casa) &
            (db_storico['AwayTeam'] == trasf) &
            ((db_storico['Data_DT'] - data_dt).abs().dt.days <= 3)
        ]
        if match_db.empty:
            continue
        rec = match_db.tail(1).to_dict(orient='records').pop(0)
        mkt = str(df_tr.loc[idx, 'Mercato'])
        p_close = p_close_pulita(rec, mkt)
        q_close_raw = rec.get(col_q_close.get(mkt, ''), np.nan)
        if pd.isna(p_close) or pd.isna(q_close_raw) or float(q_close_raw) <= 1.0:
            continue

        hg, ag = float(rec['FTHG']), float(rec['FTAG'])
        stake_v = float(df_tr.loc[idx, 'Puntata_Euro'])
        q_presa = float(df_tr.loc[idx, 'Quota_Presa'])

        vinta = {'1': hg > ag, 'X': hg == ag, '2': hg < ag, 'OVER 2.5': (hg + ag) > 2.5, 'UNDER 2.5': (hg + ag) < 2.5}[mkt]
        esito_fin = "Vinta" if vinta else "Persa"
        prof_v = round(stake_v * (q_presa - 1.0), 2) if vinta else -stake_v
        clv_net = round(((q_presa * p_close) - 1.0) * 100.0, 2)

        df_tr.loc[idx, 'Esito'] = esito_fin
        df_tr.loc[idx, 'Quota_Chiusura'] = round(float(q_close_raw), 2)
        df_tr.loc[idx, 'Profitto_Euro'] = prof_v
        df_tr.loc[idx, 'CLV_Netto_Pct'] = clv_net
        chiuse_count += 1

    return df_tr, chiuse_count


# BLOCCO 3: Interfaccia principale, barra laterale e Scheda 1 (Pronostici & Registrazione)
st.title("⚽ Prono Mel v1 — Doppio Motore Quantitativo & Profit Tracker")
st.caption("Motore A (Logit-Offset Mercato) + Motore B (Poisson Dixon-Coles Proxy xG) + Google Sheets")

with st.sidebar:
    st.header("⚙️ Parametri Operativi")
    scelta_motori = st.selectbox(
        "Motore di Selezione",
        ["Entrambi (Logit + Poisson)", "Solo Motore A (Logit-Offset)", "Solo Motore B (Poisson Dixon-Coles)"],
        index=0
    )
    budget_totale = st.number_input("Capitale di Riferimento (€)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0)
    modalita_operativa = st.radio(
        "Modalità Staking (Motore A)",
        ["Paper Trading (1% = 10€)", "Pilota Reale Micro (0.5% = 5€)"],
        index=0
    )
    flat_stake = round(budget_totale * (0.01 if "Paper" in modalita_operativa else 0.005), 2)
    st.info(f"Motore A Flat: **{flat_stake:.2f} €** (Cap {budget_totale * 0.02:.2f} €) | Motore B Flat: **5.00 €** + **1/4 Kelly**")

    soglia_ev_pct = st.slider("Soglia Minima EV Motore A (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    solo_top5 = st.checkbox("Concentra sui Top 5 Campionati (Consigliato)", value=True)

    st.divider()
    st.subheader("🔑 Line Shopping Live (Opzionale)")
    odds_api_key = st.text_input("API Key The Odds API (500 req/mese gratis)", value=st.secrets.get("odds_api_key", ""), type="password")

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
    st.subheader("Analisi Palinsesto Settimanale (Ordinato per Data e Ora ITA)")
    if st.button("🚀 Analizza Partite Future", type="primary"):
        df_seg, msg = analizza_palinsesto_futuro(
            db, elo_ratings, ultima_data, ultima_div, feat_cols, medie_feat, scaler, modelli,
            soglia_ev=soglia_ev_pct / 100.0, flat_stake=flat_stake, budget_tot=budget_totale,
            solo_top5=solo_top5, modalita_motore=scelta_motori
        )
        if not df_seg.empty:
            gia_gestite = set()
            df_tr_es = carica_tracker()
            if not df_tr_es.empty:
                gia_gestite.update(zip(df_tr_es['Partita'].astype(str), df_tr_es['Mercato'].astype(str)))
            conn_sc = get_gsheets_conn()
            if conn_sc is not None:
                try:
                    df_sc_es = conn_sc.read(worksheet="Scarti", ttl=0)
                    if df_sc_es is not None and not df_sc_es.empty:
                        gia_gestite.update(zip(df_sc_es['Partita'].astype(str), df_sc_es['Mercato'].astype(str)))
                except Exception:
                    pass
            df_seg = df_seg[~df_seg.apply(lambda r: (str(r['Partita']), str(r['Mercato'])) in gia_gestite, axis=1)].reset_index(drop=True)
        st.session_state['segnali_correnti'] = df_seg
        st.session_state['msg_segnali'] = msg


        st.session_state['segnali_correnti'] = df_seg
        st.session_state['msg_segnali'] = msg

    if 'segnali_correnti' in st.session_state:
        df_seg = st.session_state['segnali_correnti']
        if df_seg.empty:
            st.warning(st.session_state.get('msg_segnali', 'Nessun segnale trovato o tutte le partite sono già state gestite.'))
        else:
            cols_mostra = [c for c in ['Motore', 'Data', 'Ora_ITA', 'Campionato', 'Partita', 'Mercato', 'Quota_B365', 'Quota_Media', 'Quota_MAX', 'Prob_AI_Pct', 'EV_Pct', 'Puntata_Consigliata'] if c in df_seg.columns]
            st.dataframe(df_seg[cols_mostra], use_container_width=True)

            if st.button("📲 Invia Pronostici su Telegram"):
                ok_tg, msg_tg = invia_alert_telegram(df_seg)
                if ok_tg:
                    st.success(msg_tg)
                else:
                    st.warning(msg_tg)

            st.divider()
            st.subheader("🔍 Verifica Quota Live, Consenso Mercato e Registrazione")
            opzioni_match = [f"[{r['Motore']}] {r['Data']} {r['Ora_ITA']} | {r['Partita']} [{r['Mercato']}] @ {r['Quota_MAX']}" for _, r in df_seg.iterrows()]
            scelta_idx = st.selectbox("Seleziona la partita da verificare:", range(len(opzioni_match)), format_func=lambda x: opzioni_match[x])
            riga = df_seg.iloc[scelta_idx]

            col_a, col_b = st.columns(2)
            prob_base_usata = float(riga['Prob_AI_Pct'])
            usa_consenso_live = False

            with col_a:
                st.markdown("**A. Comparatore Quote Live & Andamento Mercato (The Odds API)**")
                if "cache_odds_live" not in st.session_state:
                    st.session_state["cache_odds_live"] = {}
                chiave_match = f"{riga['Motore']}_{riga['Partita']}_{riga['Mercato']}"
                if st.button("🔎 Cerca Quota Migliore & Consenso Live"):
                    df_live, p_live_ai, esito_api = cerca_migliore_quota_live(odds_api_key, riga['Sport_Key'], riga['Casa'], riga['Trasferta'], riga)
                    st.session_state["cache_odds_live"][chiave_match] = (df_live, p_live_ai, esito_api)
                if chiave_match in st.session_state["cache_odds_live"]:
                    df_live_saved, p_live_saved, esito_saved = st.session_state["cache_odds_live"][chiave_match]
                    if df_live_saved is not None:
                        st.dataframe(df_live_saved, use_container_width=True)
                        if p_live_saved is not None:
                            delta_p = p_live_saved - float(riga['Prob_AI_Pct'])
                            st.info(f"📡 **Consenso Live (effetto grafico quote):** Probabilità AI attuale = **{p_live_saved:.2f}%** ({delta_p:+.2f}% rispetto all'apertura {riga['Prob_AI_Pct']:.2f}%)")
                            usa_consenso_live = st.checkbox("Usa Prior Consenso Live (consigliato se le quote si sono mosse)", value=True)
                            if usa_consenso_live:
                                prob_base_usata = float(p_live_saved)
                    else:
                        st.info(esito_saved)

            with col_b:
                st.markdown("**B. Check Assenze, Kelly 1/4 & Calcolo EV Finale**")
                malus_pct = st.slider("Malus Assenze / Infortuni Extra (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
                prob_corretta = max(1.0, min(99.0, prob_base_usata + malus_pct))
                book_scelto = st.text_input("Bookmaker / Exchange scelto", value="Miglior Quota ADM / Exchange")
                quota_reale = st.number_input("Quota reale trovata", min_value=1.01, max_value=20.0, value=float(riga['Quota_MAX']), step=0.01)

                p_dec = prob_corretta / 100.0
                ev_finale = ((p_dec * quota_reale) - 1.0) * 100.0
                f_kelly = max(0.0, (p_dec * quota_reale - 1.0) / max(0.01, quota_reale - 1.0))
                kelly_quarto_pct = min(5.0, 0.25 * f_kelly * 100.0)
                kelly_euro = round(budget_totale * (kelly_quarto_pct / 100.0), 2)
                st.caption(f"📐 Suggerimento Kelly Frazionario (1/4): **{kelly_quarto_pct:.2f} U** (pari a **{kelly_euro:.2f} €** su {budget_totale:.0f} €)")

                puntata_default = max(1.0, round(float(riga['Puntata_Consigliata']) * 2.0) / 2.0)
                puntata_reale = st.number_input("Puntata reale (€) [arrotondata per bookmaker]", min_value=1.0, max_value=1000.0, value=puntata_default, step=0.50)

                soglia_attiva = 9.0 if riga['Motore'] == 'Poisson-xG' else soglia_ev_pct
                if riga['Motore'] == 'Poisson-xG':
                    variante = "Motore B - Poisson xG"
                elif malus_pct != 0.0:
                    variante = "Motore A - Intervento Manuale"
                elif usa_consenso_live:
                    variante = "Motore A - Consenso Live"
                else:
                    variante = "Motore A - Base Congelato"

                if ev_finale >= soglia_attiva:
                    st.success(f"Probabilità Finale: **{prob_corretta:.1f}%** | EV Reale: **+{ev_finale:.2f}%** (Giocata Valida)")
                else:
                    st.error(f"Probabilità Finale: **{prob_corretta:.1f}%** | EV Reale: **{ev_finale:.2f}%** (Sotto soglia {soglia_attiva:.1f}%: scartare)")

                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    if st.button("✅ Registra nel Tracker", type="primary"):
                        df_tr = carica_tracker()
                        nuovo_id = int(pd.to_numeric(df_tr['ID'], errors='coerce').max() + 1) if not df_tr.empty else 1
                        ora_italia_log = pd.Timestamp.now(tz="Europe/Rome").strftime("%Y-%m-%d %H:%M")
                        nuova_riga = {
                            'ID': nuovo_id, 'Data_Log': ora_italia_log, 'Data_Match': riga['Data'],
                            'Campionato': riga['Campionato'], 'Tier': riga['Tier'], 'Partita': riga['Partita'],
                            'Mercato': riga['Mercato'], 'Variante_Modello': variante, 'Bookmaker': book_scelto,
                            'Quota_Presa': quota_reale, 'Quota_Chiusura': np.nan,
                            'Prob_AI_Pct': prob_corretta, 'Malus_Assenze_Pct': malus_pct,
                            'Puntata_Euro': puntata_reale, 'Esito': 'In Corso',
                            'Profitto_Euro': 0.0, 'CLV_Netto_Pct': np.nan
                        }
                        df_tr = pd.concat([df_tr, pd.DataFrame([nuova_riga])], ignore_index=True)
                        salva_tracker(df_tr, budget_base=budget_totale)
                        st.session_state['segnali_correnti'] = df_seg.drop(df_seg.index[scelta_idx]).reset_index(drop=True)
                        st.rerun()

                with c_btn2:
                    motivo_scarto = st.text_input("Motivo eventuale scarto", value="Quota scesa / Assenze chiave")
                    if st.button("🗑️ Scarta e Archivia Decisione"):
                        registra_scarto({
                            'Data_Log': pd.Timestamp.now(tz="Europe/Rome").strftime("%Y-%m-%d %H:%M"),
                            'Partita': riga['Partita'], 'Mercato': riga['Mercato'],
                            'Quota_Rif': quota_reale, 'EV_Finale_Pct': round(ev_finale, 2),
                            'Motivo': motivo_scarto
                        })
                        st.session_state['segnali_correnti'] = df_seg.drop(df_seg.index[scelta_idx]).reset_index(drop=True)
                        st.rerun()


# BLOCCO 4: Scheda 2 (Metriche, Semaforo, Grafico Equity multi-strategia e Confronto Diretto)
with tab_tracker:
    st.subheader("📊 Profit Tracker & Semaforo Validazione")
    df_tr = carica_tracker()

    if df_tr.empty:
        st.info("Nessuna giocata ancora registrata. Usa la Scheda 1 per salvare le tue prime giocate in Paper Trading.")
    else:
        filtro_var = st.selectbox("Filtra per Strategia / Variante (Separazione A/B/C):", ["Tutte"] + sorted(df_tr['Variante_Modello'].dropna().unique().tolist()))
        df_view = df_tr if filtro_var == "Tutte" else df_tr[df_tr['Variante_Modello'] == filtro_var].copy()

        chiuse = df_view[df_view['Esito'].isin(['Vinta', 'Persa'])].copy()
        n_chiuse = len(chiuse)
        prof_tot = float(pd.to_numeric(chiuse['Profitto_Euro'], errors='coerce').sum()) if n_chiuse > 0 else 0.0
        vol_tot = float(pd.to_numeric(chiuse['Puntata_Euro'], errors='coerce').sum()) if n_chiuse > 0 else 0.0
        roi_pct = (prof_tot / vol_tot * 100.0) if vol_tot > 0 else 0.0
        clv_s = pd.to_numeric(chiuse['CLV_Netto_Pct'], errors='coerce').dropna()
        n_clv_valide = len(clv_s)
        clv_medio = float(clv_s.mean()) if n_clv_valide > 0 else 0.0

        equity = budget_totale + pd.to_numeric(chiuse['Profitto_Euro'], errors='coerce').fillna(0.0).cumsum()
        picco = np.maximum.accumulate(np.concatenate([[budget_totale], equity.values]))
        dd_pct = float(np.max((picco - np.concatenate([[budget_totale], equity.values])) / picco) * 100.0) if n_chiuse > 0 else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Giocate con CLV", f"{n_clv_valide} / 300")
        m2.metric("Profitto Netto", f"{prof_tot:+.2f} €")
        m3.metric("ROI Reale", f"{roi_pct:+.2f}%")
        m4.metric("CLV Netto Medio", f"{clv_medio:+.2f}%")
        m5.metric("Max Drawdown", f"{dd_pct:.2f}%")

        st.progress(min(1.0, n_clv_valide / 300.0), text=f"Avanzamento Validazione Paper Trading: {n_clv_valide} su 300 giocate con CLV valido")

        if dd_pct >= 15.0:
            st.error("🚨 STOP-LOSS SCATTATO: Drawdown >= 15%. Sospendere le giocate e rivedere i parametri!")
        elif n_clv_valide >= 300 and clv_medio > 0 and dd_pct < 15.0:
            st.success("🟢 SEMAFORO VERDE: Criteri di promozione raggiunti! Puoi avviare il Pilota Reale Micro (0.5%).")
        else:
            st.warning("🟡 SEMAFORO GIALLO: Continua il Paper Trading fino a 300 giocate con CLV netto positivo.")

        df_comp, df_eq = genera_tabelle_comparative(df_tr, budget_base=budget_totale)
        if not df_eq.empty and len(df_eq) > 1:
            st.markdown("### 📈 Andamento Curve Capitale (Motore A vs Motore B vs Totale)")
            st.line_chart(df_eq.set_index('Numero_Giocata')[['Capitale_Motore_A_Euro', 'Capitale_Motore_B_Euro', 'Capitale_Totale_Euro']])

        if not df_comp.empty:
            st.markdown("### ⚖️ Confronto Diretto Rendimenti tra Strategie (Sincronizzato su Google Sheets)")
            st.dataframe(df_comp, use_container_width=True)

        st.divider()
        st.markdown("### ✏️ Aggiorna Esiti e Quota di Chiusura")
        if st.button("🔄 Chiudi Partite Concluse in Automatico (da Football-Data)", type="primary"):
            st.cache_resource.clear()
            db_agg = list(prepara_motore_predittivo()).pop(0)
            df_tr_agg, n_agg = chiudi_partite_automatico(df_tr.copy(), db_agg)
            if n_agg > 0:
                salva_tracker(df_tr_agg, budget_base=budget_totale)
                st.success(f"Aggiornate automaticamente {n_agg} partite e sincronizzate le schede Google Sheets!")
                st.rerun()
            else:
                st.info("Nessuna nuova partita conclusa con quote di chiusura pubblicate trovata nei CSV.")

        if st.button("📤 Sincronizza Schede Confronto & Grafico su Google Sheets"):
            salva_tracker(df_tr, budget_base=budget_totale)
            st.success("Schede Tracker, Confronto_Strategie e Curva_Equity aggiornate su Google Sheets!")

        in_corso = df_tr[df_tr['Esito'] == 'In Corso']


        if not in_corso.empty:
            id_sel = st.selectbox("Seleziona ID giocata da chiudere:", in_corso['ID'].tolist())
            r_sel = df_tr[df_tr['ID'] == id_sel].to_dict(orient='records').pop(0)

            st.write(f"**{r_sel['Partita']}** — Mercato: **{r_sel['Mercato']}** @ **{r_sel['Quota_Presa']}** (Puntata: {r_sel['Puntata_Euro']} €)")
            c_e1, c_e2, c_e3 = st.columns(3)
            with c_e1:
                nuovo_esito = st.selectbox("Esito Finale", ["Vinta", "Persa", "Rimborsata"])
            with c_e2:
                q_chiusura = st.number_input("Quota Chiusura Reale (al calcio d'inizio)", min_value=1.01, max_value=25.0, value=float(r_sel['Quota_Presa']), step=0.01)
            with c_e3:
                aggio_chiusura = st.number_input("Aggio stimato chiusura (%)", min_value=0.0, max_value=10.0, value=4.5, step=0.5)

            if st.button("💾 Salva Chiusura Giocata"):
                idx_row = df_tr[df_tr['ID'] == id_sel].index.tolist().pop(0)
                stake_v = float(df_tr.loc[idx_row, 'Puntata_Euro'])
                q_presa = float(df_tr.loc[idx_row, 'Quota_Presa'])
                prof_v = round(stake_v * (q_presa - 1.0), 2) if nuovo_esito == "Vinta" else (-stake_v if nuovo_esito == "Persa" else 0.0)
                p_close_clean = (1.0 / q_chiusura) / (1.0 + aggio_chiusura / 100.0)
                clv_net = round(((q_presa * p_close_clean) - 1.0) * 100.0, 2)

                df_tr.loc[idx_row, 'Esito'] = nuovo_esito
                df_tr.loc[idx_row, 'Quota_Chiusura'] = q_chiusura
                df_tr.loc[idx_row, 'Profitto_Euro'] = prof_v
                df_tr.loc[idx_row, 'CLV_Netto_Pct'] = clv_net
                salva_tracker(df_tr, budget_base=budget_totale)
                st.success(f"Esito salvato! Profitto: {prof_v:+.2f} € | CLV Netto: {clv_net:+.2f}%")
                st.rerun()

        st.markdown("### 📋 Registro Completo Giocate")
        st.dataframe(df_view.sort_values(by='ID', ascending=False), use_container_width=True)


# BLOCCO 5: Scheda 3 (Guida operativa completa integrata nell'applicazione)
with tab_guida:
    st.subheader("📘 Manuale Operativo Completo — Prono Mel v1 (Doppio Motore)")
    st.markdown("""
    ### 1. I Due Motori di Selezione Integrati
    * **Motore A (`Logit-Offset Mercato`):** Parte dalle quote depurate con *Power Devig* su tutti i campionati selezionati, corregge *Outcome Bias*, tiri in porta ed *Elo*. Range quote `1.35 – 4.00`, soglia `EV >= +3.0%`.
    * **Motore B (`Poisson Dixon-Coles / Proxy xG`):** Analizza **Serie A, Premier League, Bundesliga e La Liga** (dalla 5ª giornata in poi) costruendo la matrice `7×7` dei risultati esatti su forza offensiva (`AS`) e difensiva (`DW`) pesata (`70%` forma recente / `30%` stagione). Range quote **`1.65 – 2.40`**, soglia **`EV >= +9.0%` e `<= +22.0%`** (anti-anomalia), stake fissa **`5,00 €`** + indicatore **`1/4 Kelly`**.

    ### 2. Quando scommettere: Anticipo vs T-45 minuti & Consenso Live
    * **Giocata in Anticipo (Giovedì sera / Venerdì):** Se trovi una quota alta nel comparatore live senza dubbi di infortuni o coppe, **bloccare subito la quota** massimizza il **CLV Netto** prima che il mercato la faccia scendere.
    * **Check Pre-Partita (Sabato e Domenica a T-45 min):** L'alert Telegram ti avvisa 45 minuti prima delle sole partite estratte (ordinate per Data e Ora ITA).
    * **Consenso Live & Malus Assenze (%):** Lascia attiva la spunta **Usa Prior Consenso Live** (per il Motore A) e lascia il cursore **Malus Assenze** a **`0.00`** nella quasi totalità dei casi: il consenso live incorpora già i movimenti del grafico quote (es. Oddspedia), evitandoti "EV finti".

    ### 3. Gestione Lista, Google Sheets & Grafici
    * **Pulizia automatica tabella:** Ogni partita registrata o scartata sparisce subito dall'elenco delle partite da valutare.
    * **Chiusura Automatica (Lunedì alle 10:45):** Premi **🔄 Chiudi Partite Concluse in Automatico** nella Scheda 2. Il software calcola il **CLV Netto tramite Power Devig reale** sulle quote di chiusura ufficiali e aggiorna automaticamente su Google Sheets sia **`Confronto_Strategie`** sia **`Curva_Equity`**.
    * **Semaforo Validazione:** **🟡 Giallo** fino a **300 giocate con CLV valido** -> **🟢 Verde** se `CLV Netto Medio > 0%` e `Max Drawdown < 15%`.
    """)
