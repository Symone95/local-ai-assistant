import json
import os
import random
import time

try:
	import vlc
except ImportError:
	vlc = None


class RadioManager:

	def __init__(self, risorse_path=None, vlc_instance=None):
		self.risorse_path = risorse_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		self.radio_json_path = os.path.join(self.risorse_path, "dto", "risorse", "radio", "stazioni_radio.json")
		if not os.path.isfile(self.radio_json_path):
			self.radio_json_path = os.path.join(self.risorse_path, "risorse", "radio", "stazioni_radio.json")

		self.vlc_instance = vlc_instance
		if self.vlc_instance is None and vlc is not None:
			self.vlc_instance = vlc.Instance()
		self.player = None
		if self.vlc_instance is not None:
			self.player = self.vlc_instance.media_player_new()

		self.set_status(True)
		self.current_audio = ""  # Salvo l'audio in ascolto
		self.media = None
		self.stazioni_radio_salvate = [x for x in json.load(open(self.radio_json_path, "r", encoding="utf-8"))]

	def set_status(self, status):
		"""Starto e stoppo la radio in base allo status True e False"""
		self.status = status

	def play_radio(self, canzone, dashboard_radio_layout=None, current_radio_label=None, btn=None):
		"""
		Avvia la canzone selezionata, se è un link allora leggo da remoto
		"""

		if self.vlc_instance is None or self.player is None:
			raise RuntimeError("VLC non è disponibile. Installa python-vlc e assicurati che il modulo sia importabile.")

		# Riabilito la dashboard che indica la radio selezionata e il bottone per stop e resume
		if dashboard_radio_layout:  # Controllo per eventuale avvio audio locale
			dashboard_radio_layout.opacity = 1
			dashboard_radio_layout.disabled = False

		if not isinstance(canzone, str):
			canzone = canzone.text

		self.media = self.vlc_instance.media_new(canzone)

		self.media.get_mrl()
		self.player.set_media(self.media)

		self.player.play()
		self.current_audio = canzone
		if current_radio_label:  # Controllo per eventuale avvio audio locale
			current_radio_label.text = canzone
		time.sleep(1.5)  # startup time.
		if canzone.startswith("http"):
			# Streaming from url
			print("Leggo da url: %s" % canzone)
		else:
			duration = self.player.get_length() / 1000  # Lunghezza in millisecondi, divido per 100 per trovare la lunghezza in secondi
			print("Durata canzone: %s secondi" % duration)
			mm, ss = divmod(duration, 60)
			print("Canzone selezionata: %s - Lunghezza: %02d:%02d minuti" % (canzone, mm, ss))

		print("RADIO STATUS INTERNO: %s" % self.status)
		# Finchè continua a leggere ascolto a meno che lo status cambia a False per stopparlo

		# while self.player.is_playing():  # and self.status:
			#print("Tempo: %s" % player.get_time() if player.get_time() is not None else "tempo non trovato")
			#print("Titolo: %s" % player.get_title() if player.get_title() is not None else "titolo non trovato" )
			#print("audio_get_volume: %s" % player.audio_get_volume() if player.audio_get_volume() is not None else "audio track description non trovata" )
			#print("Canale audio: %s" % player.audio_get_channel() if player.audio_get_channel() is not None else "audio channel non trovato" )
			#print("Position: %s" % player.get_position() if player.get_position() is not None else "Position non trovato" )
			#print(dir(player))
			#print("SONO NEL CICLO")

			#if not self.status:
			#	break
			#time.sleep(1)

		#time.sleep(duration)
		print("Finished playing your song %s" % canzone)
		return self.player

	def leggi_radio_info_json(self):
		return self.stazioni_radio_salvate

	def cerca_radio(self, nome_radio):
		lista_radio = self.leggi_radio_info_json()
		for radio in lista_radio:
			if radio["nome"].lower() == nome_radio.lower():
				return radio
		return None

	def get_station_list(self):
		return self.stazioni_radio_salvate

	def random_radio(self):
		"""
		Funzione per selezionare una stazione radio casualmente
		:return:
		"""
		random_radio = self.stazioni_radio_salvate[random.randint(0, len(self.stazioni_radio_salvate) - 1)]
		self.play_radio(random_radio["url"])

	def stop_radio(self):
		if self.player is None:
			raise RuntimeError("VLC non è disponibile. Installa python-vlc e assicurati che il modulo sia importabile.")
		if self.player.is_playing():
			self.player.stop()
			self.set_status(False)
			self.current_audio = ""
			self.media = None
			return True
		return False

	def is_playing(self):
		if self.player is None:
			return False
		return bool(self.player.is_playing())

	def stop_resume_radio(self, stop_resume_radio_button, btn):
		if self.player.is_playing():
			stop_resume_radio_button.text = "Resume"
			self.player.stop()
		else:
			stop_resume_radio_button.text = "Stop"
			self.player.play()


radio_manager = RadioManager()
