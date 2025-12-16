# SporttiAppi

Soveltava matematiikka ja fysiikka ohjelmoinnissa -opintojakson projektityö.

## Kuvaus

Streamlit-sovellus, joka analysoi puhelimella kerättyä liikuntadataa (GPS-sijainti ja kiihtyvyysanturi). Sovellus laskee ja visualisoi:

- **Askelmäärän** kahdella eri menetelmällä:
  - Nollaylitysmenetelmä (suodatettu kiihtyvyysdata)
  - Fourier-analyysi (tehospektri)
- **Kuljetun matkan** Haversinen kaavalla
- **Keskinopeuden** ja **askelpituuden**
- **Reitin kartalla** (Folium)

## Käytetyt menetelmät

- **Butterworth-suodatin**: Alipäästösuodatin kiihtyvyysdatan tasoittamiseen
- **Haversinen kaava**: Etäisyyden laskeminen GPS-koordinaattien välillä
- **Fourier-analyysi**: Dominoivan taajuuden (askeltaajuus) määrittäminen

## Käyttö

```
streamlit run https://raw.githubusercontent.com/villepekkaa/SporttiAppi-Python/main/main.py
```

## Riippuvuudet

- streamlit
- pandas
- numpy
- matplotlib
- scipy
- folium
- streamlit-folium

## Data

Sovellus lataa datan automaattisesti GitHubista:
- `Location.csv` - GPS-sijaintidata
- `Linear Accelerometer.csv` - Kiihtyvyysanturidata
