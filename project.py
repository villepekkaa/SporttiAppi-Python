import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ladataan data suoraan GitHubista
github_base_url = 'https://raw.githubusercontent.com/villepekkaa/SporttiAppi-Python/main/data/'
df_location_data = pd.read_csv(github_base_url + 'Location.csv')
df_accelerometer_data = pd.read_csv(github_base_url + 'Linear Accelerometer.csv')

#Annetaan visualisoinnille otsikko
st.title('Aamureippailu')

#Tuodaan filtterifunktiot.
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff,  nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


# Suodatetaan ensimmäiset 20 sekuntia pois
df = df_location_data[df_location_data['Time (s)'] > 20]
df = df.reset_index(drop=True)

# Asetetaan aika alkamaan nollasta
df['Time (s)'] = df['Time (s)'] - df['Time (s)'].min()

#Lasketaan matka käyttäen Haversinen kaava
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

#Data suodatetaan alipäästösuodattimella, joka poistaa siitä valittua cut-off -taajuutta suuremmilla
#taajuuksilla tapahtuvat vaihtelut.
#Käytännössä dataa "tasoitetaan", eli alipäästösuodatin vastaa jossain määrin liukuvaa keskiarvoa.

# Suodatetaan ensimmäiset 20 sekuntia pois kiihtyvyysdatasta
df_accel = df_accelerometer_data[df_accelerometer_data['Time (s)'] > 20].copy()
df_accel = df_accel.reset_index(drop=True)
df_accel['Time (s)'] = df_accel['Time (s)'] - df_accel['Time (s)'].min()

data = df_accel['Y (m/s^2)']
T_tot = df_accel['Time (s)'].max() #Koko datan pituus
n = len(df_accel['Time (s)']) #Datapisteiden lukumäärä
fs = n/T_tot #Näytteenottotaajus, OLETETAAN VAKIOKSI
nyq = fs/2 #Nyqvistin taajuus, suurin taajuus, joka datasta voidaan havaita
order = 3
cutoff = 1 / 0.1  #Cut-off taajuus, tätä suuremmat taajuuden alipäästösuodatin poistaa datasta
#Cut-off -taajuuden tulee olla riittävän pieni, jotta data yleensäkin suodattuu
#Cut-off -taajuuden ei tule olla niin pieni, että se suodattaisi pois askelia
data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

#Piirretään kuvaa, jossa alkuperäinen ja suodatettu signaali
st.subheader('Suodatetun kiihtyvyysdatan Y-komponentti')
fig1, ax1 = plt.subplots(figsize=(12,4))
plt.plot(df_accel['Time (s)'],data,label = 'data')
plt.plot(df_accel['Time (s)'],data_filt,label = 'suodatettu data')
plt.axis([0,60,-10,10])
plt.grid()
plt.legend()
st.pyplot(fig1)

#Lasketaan askeleet
#Tutkitaan, kuinka usein suodatettu signaali ylittää nollatason
jaksot = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0: #True jos arvoilla data_filt[i] ja data_filt[i+1] on eri etumerkki --> nollan ylitys
        jaksot = jaksot + 1/2

# Fourier-analyysi tehospektrin laskemiseksi
st.subheader('Tehospektri (Fourier-analyysi)')
signal = df_accel['Y (m/s^2)']
t = df_accel['Time (s)']
N = len(signal)
dt = np.max(t/N)

# Fourier muunnos
fourier = np.fft.fft(signal, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)
L = np.arange(1, int(N/2))

# Piirretään tehospektri
fig2, ax2 = plt.subplots(figsize=(15, 6))
plt.plot(freq[L], psd[L].real)
plt.xlabel('Taajuus (Hz)')
plt.ylabel('Teho')
plt.axis([0,10,0,20000])
plt.grid()
st.pyplot(fig2)

# Näytetään dominoiva taajuus
f_max = freq[L][psd[L] == np.max(psd[L])][0]
T = 1/f_max
steps_fourier = f_max * np.max(t)

st.subheader('Askelmäärät eri menetelmillä')
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Nollaylitysmenetelmä (suodatettu data)", value=f"{int(jaksot)} askelta")
    st.caption("Laskettu suodatetun signaalin nollakohtien perusteella")
with col2:
    st.metric(label="Fourier-analyysi", value=f"{int(steps_fourier)} askelta")
    st.caption(f"Dominoiva taajuus: {f_max:.2f} Hz | Jaksonaika: {T:.2f} s")

#Lasketaan kuljettu matka
import numpy as np
df['Distance_calc'] = np.zeros(len(df))

#lasketaan väimatka havaintopisteiden välillä käyttäen For-luuppia
for i in range(len(df)-1):
    lon1 = df['Longitude (°)'][i]
    lon2 = df['Longitude (°)'][i+1]
    lat1 = df['Latitude (°)'][i]
    lat2 = df['Latitude (°)'][i+1]
    df.loc[i+1,'Distance_calc'] = haversine(lon1, lat1, lon2, lat2)

#Lasketaan kokonaismatka mittapisteiden välisestä matkasta
df['total_distance'] = df['Distance_calc'].cumsum()

# Lasketaan keskinopeus ja askelpituus
kokonaismatka = df['total_distance'].max()  
kokonaisaika = df['Time (s)'].max()  
askelmäärä = jaksot  

# Keskinopeus
keskinopeus_km_h = (kokonaismatka / kokonaisaika) * 3600  # km/h

# Askelpituus
askelpituus = (kokonaismatka * 1000) / askelmäärä  # metriä

# Näytetään tilastot Streamlitissä
st.subheader('Harjoituksen tilastot')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Matka", value=f"{kokonaismatka:.3f} km")
with col2:
    st.metric(label="Keskinopeus", value=f"{keskinopeus_km_h:.2f} km/h")

with col3:
    st.metric(label="Askelpituus", value=f"{askelpituus:.2f} m")
with col4:
    st.metric(label="Kesto", value=f"{kokonaisaika/60:.1f} min")
    

st.subheader('Matka ajan funktiona')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,5))
plt.plot(df['Time (s)'],df['total_distance'])
plt.ylabel('Kokonaismatka [km]')
plt.xlabel('Aika [s]')
st.pyplot(fig)

#Kartta
st.subheader('Kuljettu reitti kartalla')
import folium
from streamlit_folium import st_folium

#Määritellään kartan keskipiste
lat1 = df['Latitude (°)'].mean()
long1 = df['Longitude (°)'].mean()

#luodaan kartta
my_map = folium.Map(location=[lat1, long1], zoom_start=15)

#Piirretään reitti kartalle
route = folium.PolyLine(df[['Latitude (°)','Longitude (°)']], color='red', weight=3)
route.add_to(my_map)

#Zoomaa kartta reitin mukaan
my_map.fit_bounds(route.get_bounds())

st_map = st_folium(my_map, width=900, height=650)