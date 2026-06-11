import time

import requests


class MeteoManager:

    def __init__(self):
        self.APPID = "OPENWEATHER_API_KEY_RIMOSSA"
        self.URL_BASE = "http://api.openweathermap.org/data/2.5/"

    def current_weather(self, q: str = "") -> dict:
        """https://openweathermap.org/api"""
        if not q:
            return {"cod": "400", "message": "Nessuna città specificata."}

        ris = requests.get(self.URL_BASE + f"weather?q={q}&lang=it&units=metric&APPID={self.APPID}").json()

        if ris["cod"] == "404":
            return ris
        else:
            return {
                    "descrizione": ris["weather"][0]["description"],
                    "Umidità": str(ris["main"]["humidity"]) + "%",
                    "Temperatura": str(ris["main"]["temp"]) + "°C",
                    "Temperatura minima": str(ris["main"]["temp_min"]) + "°C",
                    "Temperatura massima": str(ris["main"]["temp_max"]) + "°C",
                    "Pressione": str(ris["main"]["pressure"]) + "hPa",
                    "Velocità del vento": str(ris["wind"]["speed"]) + " metre/sec",
                    "Alba": time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(ris["sys"]["sunrise"])),
                    "Tramonto": time.strftime('%d-%m-%Y %H:%M:%S', time.localtime(ris["sys"]["sunset"])),
                    "Coordinate": (ris["coord"]["lat"], ris["coord"]["lon"])
                }


meteo_manager = MeteoManager()