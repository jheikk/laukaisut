# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:12:30 2025

@author: OMISTAJA
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Streamlit-sovelluksen konfiguraatio
st.set_page_config(
    page_title="Valioliiga Laukausennustus",
    page_icon="⚽",
    layout="wide"
)

def get_file_info():
    """Hakee CSV-tiedoston aikaleiman ja koon"""
    try:
        stat = os.stat('valioliigadata_yksityiskohtaiset_keskiarvot.csv')
        return stat.st_mtime, stat.st_size
    except FileNotFoundError:
        return 0, 0

@st.cache_data(ttl=1800)  # Cache vanhenee 30 minuutissa
def lataa_ja_kasittele_data(file_timestamp, file_size):
    """Lataa CSV-data ja laskee keskiarvot"""
    try:
        # Ladataan historiallinen data
        historiallinen = pd.read_csv('valioliigadata_yksityiskohtaiset_keskiarvot.csv')
        
        # Lasketaan joukkueiden keskiarvot
        keskiarvot = {}
        
        for _, peli in historiallinen.iterrows():
            kotijoukkue = peli['Kotijoukkue']
            vierasjoukkue = peli['Vierasjoukkue']
            
            # Alusta joukkueet jos ei ole vielä
            if kotijoukkue not in keskiarvot:
                keskiarvot[kotijoukkue] = {
                    'koti_laukaukset': [],
                    'vieras_laukaukset': []
                }
            if vierasjoukkue not in keskiarvot:
                keskiarvot[vierasjoukkue] = {
                    'koti_laukaukset': [],
                    'vieras_laukaukset': []
                }
            
            # Lisää laukaukset (jos ei ole NaN)
            if pd.notna(peli['Kotil']):
                keskiarvot[kotijoukkue]['koti_laukaukset'].append(peli['Kotil'])
            if pd.notna(peli['Vierasl']):
                keskiarvot[vierasjoukkue]['vieras_laukaukset'].append(peli['Vierasl'])
        
        # Laske keskiarvot
        for joukkue in keskiarvot:
            koti_ka = np.mean(keskiarvot[joukkue]['koti_laukaukset']) if keskiarvot[joukkue]['koti_laukaukset'] else 0
            vieras_ka = np.mean(keskiarvot[joukkue]['vieras_laukaukset']) if keskiarvot[joukkue]['vieras_laukaukset'] else 0
            
            keskiarvot[joukkue]['koti_keskiarvo'] = koti_ka
            keskiarvot[joukkue]['vieras_keskiarvo'] = vieras_ka
            keskiarvot[joukkue]['yleinen_koti_ka'] = koti_ka
            keskiarvot[joukkue]['yleinen_vieras_ka'] = vieras_ka
        
        # Hae joukkueet
        kaikki_joukkueet = sorted(list(set(
            historiallinen['Kotijoukkue'].tolist() + 
            historiallinen['Vierasjoukkue'].tolist()
        )))
        
        return historiallinen, keskiarvot, kaikki_joukkueet
        
    except FileNotFoundError:
        st.error("❌ CSV-tiedostoa 'valioliigadata_yksityiskohtaiset_keskiarvot.csv' ei löydy!")
        st.info("Varmista, että tiedosto on samassa kansiossa kuin tämä skripti.")
        return None, None, None

@st.cache_resource
def treeni_mallit(historiallinen_data, keskiarvot):
    """Kouluttaa XGBoost-mallit"""
    
    # Valmistetaan treenidata
    features = [
        'Kotijoukkueen_kerroin', 'Tasapelikerroin', 'Vierasjoukkueen_kerroin',
        'Kotijoukkue_laukausten_ka', 'Vierasjoukkue_laukausten_ka',
        'Kotijoukkue_koti_laukausten_ka', 'Kotijoukkue_vieras_laukausten_ka',
        'Vierasjoukkue_koti_laukausten_ka', 'Vierasjoukkue_vieras_laukausten_ka'
    ]
    
    # Käsitellään joukkuenimet
    kaikki_joukkueet = list(set(
        historiallinen_data['Kotijoukkue'].tolist() + 
        historiallinen_data['Vierasjoukkue'].tolist()
    ))
    
    le_koti = LabelEncoder()
    le_vieras = LabelEncoder()
    
    le_koti.fit(kaikki_joukkueet)
    le_vieras.fit(kaikki_joukkueet)
    
    historiallinen_data['Kotijoukkue_encoded'] = le_koti.transform(historiallinen_data['Kotijoukkue'])
    historiallinen_data['Vierasjoukkue_encoded'] = le_vieras.transform(historiallinen_data['Vierasjoukkue'])
    
    features.extend(['Kotijoukkue_encoded', 'Vierasjoukkue_encoded'])
    
    # Poistetaan puuttuvat arvot
    historiallinen_clean = historiallinen_data.dropna(subset=features + ['Kotil', 'Vierasl'])
    
    # Train/test jako
    X_full = historiallinen_clean[features]
    y_koti_full = historiallinen_clean['Kotil']
    y_vieras_full = historiallinen_clean['Vierasl']
    
    X_train, X_test, y_koti_train, y_koti_test, y_vieras_train, y_vieras_test = train_test_split(
        X_full, y_koti_full, y_vieras_full, 
        test_size=0.2, 
        random_state=42
    )
    
    # Kouluta mallit
    model_koti = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model_koti.fit(X_train, y_koti_train)
    
    model_vieras = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model_vieras.fit(X_train, y_vieras_train)
    
    return model_koti, model_vieras, le_koti, le_vieras, features

def tee_ennustus(kotijoukkue, vierasjoukkue, koti_kerroin, tasa_kerroin, vieras_kerroin, 
                keskiarvot, model_koti, model_vieras, le_koti, le_vieras, features):
    """Tekee ennustuksen yhdelle pelille"""
    
    # Luo input-data
    input_data = {
        'Kotijoukkueen_kerroin': koti_kerroin,
        'Tasapelikerroin': tasa_kerroin,
        'Vierasjoukkueen_kerroin': vieras_kerroin,
        'Kotijoukkue_laukausten_ka': keskiarvot[kotijoukkue]['yleinen_koti_ka'],
        'Vierasjoukkue_laukausten_ka': keskiarvot[vierasjoukkue]['yleinen_vieras_ka'],
        'Kotijoukkue_koti_laukausten_ka': keskiarvot[kotijoukkue]['koti_keskiarvo'],
        'Kotijoukkue_vieras_laukausten_ka': keskiarvot[kotijoukkue]['vieras_keskiarvo'],
        'Vierasjoukkue_koti_laukausten_ka': keskiarvot[vierasjoukkue]['koti_keskiarvo'],
        'Vierasjoukkue_vieras_laukausten_ka': keskiarvot[vierasjoukkue]['vieras_keskiarvo'],
        'Kotijoukkue_encoded': le_koti.transform([kotijoukkue])[0],
        'Vierasjoukkue_encoded': le_vieras.transform([vierasjoukkue])[0]
    }
    
    # Luo DataFrame
    X_input = pd.DataFrame([input_data])
    X_input = X_input[features]  # Varmista oikea järjestys
    
    # Tee ennustukset
    koti_ennustus = model_koti.predict(X_input)[0]
    vieras_ennustus = model_vieras.predict(X_input)[0]
    
    return koti_ennustus, vieras_ennustus

def laske_panossuositus(yhteensa_laukaukset, raja):
    """Laskee panossuosituksen"""
    ero = yhteensa_laukaukset - raja
    
    if ero >= 6:
        return "🔥 ISO PANOS OVER!", "Iso panos OVER", ero, "success"
    elif ero >= 5:
        return "📈 3 yksikköä OVER", "3 yksikköä OVER", ero, "success"
    elif ero >= 4:
        return "📊 2 yksikköä OVER", "2 yksikköä OVER", ero, "warning"
    elif ero >= 3:
        return "📉 1 yksikkö OVER", "1 yksikkö OVER", ero, "warning"
    elif ero <= -6:
        return "🔥 ISO PANOS UNDER!", "Iso panos UNDER", ero, "success"
    elif ero <= -5:
        return "📈 3 yksikköä UNDER", "3 yksikköä UNDER", ero, "success"
    elif ero <= -4:
        return "📊 2 yksikköä UNDER", "2 yksikköä UNDER", ero, "warning"
    elif ero <= -3:
        return "📉 1 yksikkö UNDER", "1 yksikkö UNDER", ero, "warning"
    else:
        return "❌ Älä pelaa", "Ei panosta", ero, "error"

def main():
    # Otsikko
    st.title("⚽ Valioliiga Laukausennustus")
    st.subheader("XGBoost-malli vedonlyönnin tueksi")
    
    # Sivupalkki datan hallinnalle
    st.sidebar.header("🔄 Datan hallinta")
    
    # Näytä tiedoston tila
    file_timestamp, file_size = get_file_info()
    if file_timestamp > 0:
        last_modified = datetime.fromtimestamp(file_timestamp)
        st.sidebar.success(f"📄 Data ladattu: {last_modified.strftime('%d.%m.%Y %H:%M')}")
    else:
        st.sidebar.error("❌ CSV-tiedostoa ei löydy")
    
    # Päivitä data -nappi
    if st.sidebar.button("🆕 Päivitä data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Lataa data
    with st.spinner("Ladataan dataa..."):
        historiallinen, keskiarvot, joukkueet = lataa_ja_kasittele_data(file_timestamp, file_size)
    
    if historiallinen is None:
        st.stop()
    
    # Treeni mallit
    with st.spinner("Koulutetaan koneoppimismallit..."):
        model_koti, model_vieras, le_koti, le_vieras, features = treeni_mallit(historiallinen, keskiarvot)
    
    st.success("✅ Mallit koulutettu onnistuneesti!")
    
    # Sivupalkki syötteille
    st.sidebar.header("🔧 Pelin tiedot")
    
    # Joukkueiden valinta
    kotijoukkue = st.sidebar.selectbox(
        "Kotijoukkue:",
        joukkueet,
        index=0
    )
    
    vierasjoukkue = st.sidebar.selectbox(
        "Vierasjoukkue:",
        [j for j in joukkueet if j != kotijoukkue],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Kertoimet")
    
    # Kertoimet
    col1, col2 = st.sidebar.columns(2)
    with col1:
        koti_kerroin = st.number_input(
            "Koti:", 
            min_value=1.01, 
            max_value=50.0, 
            value=2.50, 
            step=0.01
        )
        vieras_kerroin = st.number_input(
            "Vieras:", 
            min_value=1.01, 
            max_value=50.0, 
            value=2.50, 
            step=0.01
        )
    
    with col2:
        tasa_kerroin = st.number_input(
            "Tasapeli:", 
            min_value=1.01, 
            max_value=50.0, 
            value=3.50, 
            step=0.01
        )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Over/Under")
    
    # Over/Under raja
    ou_raja = st.sidebar.number_input(
        "Over/Under raja:",
        min_value=0.5,
        max_value=50.0,
        value=17.5,
        step=0.5
    )
    
    # Ennustus-nappi
    if st.sidebar.button("🚀 TEE ENNUSTUS", type="primary"):
        
        # Tee ennustus
        with st.spinner("Lasketaan ennustusta..."):
            koti_ennustus, vieras_ennustus = tee_ennustus(
                kotijoukkue, vierasjoukkue, koti_kerroin, tasa_kerroin, vieras_kerroin,
                keskiarvot, model_koti, model_vieras, le_koti, le_vieras, features
            )
        
        yhteensa = koti_ennustus + vieras_ennustus
        suositus_teksti, suositus_lyhyt, ero, alert_tyyppi = laske_panossuositus(yhteensa, ou_raja)
        
        # Näytä tulokset
        st.markdown("---")
        st.header("📊 ENNUSTUSTULOKSET")
        
        # Pelin perustiedot
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Kotijoukkue",
                f"{koti_ennustus:.1f}",
                f"laukausta"
            )
            st.caption(f"🏠 {kotijoukkue}")
        
        with col2:
            st.metric(
                "Vierasjoukkue", 
                f"{vieras_ennustus:.1f}",
                f"laukausta"
            )
            st.caption(f"✈️ {vierasjoukkue}")
        
        with col3:
            st.metric(
                "Yhteensä",
                f"{yhteensa:.1f}",
                f"{ero:+.1f} rajaan"
            )
            st.caption(f"🎯 Raja: {ou_raja}")
        
        # Panossuositus
        st.markdown("---")
        st.subheader("💡 PANOSSUOSITUS")
        
        if alert_tyyppi == "success":
            st.success(f"**{suositus_teksti}**")
        elif alert_tyyppi == "warning":
            st.warning(f"**{suositus_teksti}**")
        else:
            st.error(f"**{suositus_teksti}**")
        
        st.info(f"**Ero rajaan:** {ero:+.1f} laukausta")
        
        # Visualisointi
        st.markdown("---")
        st.subheader("📈 Visualisointi")
        
        # Pylväsdiagrammi
        fig = go.Figure(data=[
            go.Bar(name='Ennustettu', x=['Kotijoukkue', 'Vierasjoukkue'], 
                   y=[koti_ennustus, vieras_ennustus],
                   marker_color=['lightblue', 'lightcoral'])
        ])
        
        fig.add_hline(y=ou_raja, line_dash="dash", line_color="red", 
                      annotation_text=f"O/U Raja: {ou_raja}")
        
        fig.update_layout(
            title=f"{kotijoukkue} vs {vierasjoukkue} - Laukausennustus",
            yaxis_title="Laukaukset",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Joukkueiden historiatiedot
        with st.expander("📋 Joukkueiden keskiarvotiedot"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🏠 {kotijoukkue}")
                st.write(f"**Keskiarvo kotona:** {keskiarvot[kotijoukkue]['koti_keskiarvo']:.1f}")
                st.write(f"**Keskiarvo vieraissa:** {keskiarvot[kotijoukkue]['vieras_keskiarvo']:.1f}")
            
            with col2:
                st.subheader(f"✈️ {vierasjoukkue}")
                st.write(f"**Keskiarvo kotona:** {keskiarvot[vierasjoukkue]['koti_keskiarvo']:.1f}")
                st.write(f"**Keskiarvo vieraissa:** {keskiarvot[vierasjoukkue]['vieras_keskiarvo']:.1f}")
    
    # Mallin tiedot sivupalkissa
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
        **ℹ️ Tietoja mallista:**
        - XGBoost-regressio
        - Koulutettu historiallisella datalla ({len(historiallinen)} peliä)
        - Huomioi joukkueiden keskiarvot
        - Kertoimet ja koti/vieras-etu
        - Cache vanhenee 30 min välein
        """
    )

if __name__ == "__main__":
    main()
