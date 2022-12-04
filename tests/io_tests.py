import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from preprocessing.io import SnowfinchNestRecording, validate_recording_data, load_recording_data


def generate_audio(sample_rate: int, length_sec: int) -> np.ndarray:
	return np.random.random(sample_rate * length_sec) * 2.0 - 1.0


def generate_labels(start: float, end: float, count: int, labels: list[str]) -> pd.DataFrame:
	max_label_length_sec = (end - start) / count

	label_starts = np.linspace(start, end - max_label_length_sec, num = count)
	label_ends = label_starts + np.random.random(count) * max_label_length_sec
	label_indices = np.random.randint(0, len(labels), count)

	return pd.DataFrame(
		data = {
			'start': label_starts,
			'end': label_ends,
			'label': np.array(labels)[label_indices]
		}
	)


def generate_nest_recoring(
		sample_rate: int, length_sec: int, label_count: int,
		brood_age: int, brood_size: int, labels: list[str]
) -> SnowfinchNestRecording:
	audio = generate_audio(sample_rate, length_sec)
	labels = generate_labels(0.0, length_sec, label_count, labels)
	return SnowfinchNestRecording(audio, sample_rate, labels, brood_age, brood_size)


def generate_and_save_recording(
		sample_rate: int, length_sec: int, label_count: int,
		brood_age: int, brood_size: int, labels: list[str]
) -> tuple[str, str]:
	test_dir = '_data/test'
	Path(test_dir).mkdir(parents = True, exist_ok = True)

	title = f'TEST_ID-BA{brood_age}_BS{brood_size}-test'

	audio = generate_audio(sample_rate, length_sec)
	sf.write(f'{test_dir}/{title}.flac', data = audio, samplerate = sample_rate)

	labels = generate_labels(0.0, length_sec, label_count, labels)
	labels.to_csv(f'{test_dir}/{title}.txt', sep = '\t', header = False)

	return title, test_dir


class TestLoadRecordingData(unittest.TestCase):
	def setUp(self) -> None:
		self.sample_rate = 48000
		self.length_sec = 120
		self.label_count = 20
		self.brood_size = 3
		self.brood_age = 10
		self.labels = ['contact', 'feeding']

		self.rec_title, self.data_dir = generate_and_save_recording(
			sample_rate = self.sample_rate, length_sec = self.length_sec, label_count = self.label_count,
			brood_size = self.brood_size, brood_age = self.brood_age, labels = self.labels
		)

	def doCleanups(self) -> None:
		audio_file = f'{self.data_dir}/{self.rec_title}.flac'
		if Path(audio_file).exists():
			os.remove(audio_file)

		labels_file = f'{self.data_dir}/{self.rec_title}.txt'
		if Path(labels_file).exists():
			os.remove(labels_file)

		os.removedirs(self.data_dir)

	def test_obj_type_ok_for_correct_data(self):
		rec_data = load_recording_data(self.data_dir, self.rec_title)
		self.assertIsInstance(rec_data, SnowfinchNestRecording)

	def test_audio_ok_for_correct_data(self):
		rec_data = load_recording_data(self.data_dir, self.rec_title)
		self.assertIsInstance(rec_data.audio_data, np.ndarray)
		self.assertEqual(self.sample_rate, rec_data.audio_sample_rate)
		self.assertEqual(self.length_sec, rec_data.audio_data.shape[0] / rec_data.audio_sample_rate)
		self.assertTrue(all(rec_data.audio_data >= -1.0) and all(rec_data.audio_data <= 1.0))

	def test_all_labels_loaded_for_correct_data(self):
		rec_data = load_recording_data(self.data_dir, self.rec_title)
		self.assertIsInstance(rec_data.labels, pd.DataFrame)
		self.assertEqual((self.label_count, 3), rec_data.labels.shape)

	def test_value_error_for_non_nunmeric_brood_age(self):
		invalid_title = self.rec_title.replace(str(self.brood_age), 'xx')
		self.assertRaises(ValueError, lambda: load_recording_data(self.data_dir, invalid_title))

	def test_value_error_for_non_nunmeric_brood_size(self):
		invalid_title = self.rec_title.replace(str(self.brood_age), 'yy')
		self.assertRaises(ValueError, lambda: load_recording_data(self.data_dir, invalid_title))

	def test_value_error_for_invalid_filename(self):
		self.assertRaises(ValueError, lambda: load_recording_data(self.data_dir, 'invalid_title'))

	def test_error_for_missing_labels_file(self):
		os.remove(f'{self.data_dir}/{self.rec_title}.txt')
		self.assertRaises(FileNotFoundError, lambda: load_recording_data(self.data_dir, self.rec_title))

	def test_error_for_missing_audio_file(self):
		os.remove(f'{self.data_dir}/{self.rec_title}.flac')
		self.assertRaises(FileNotFoundError, lambda: load_recording_data(self.data_dir, self.rec_title))


class TestValidateRecordingData(unittest.TestCase):
	def test_no_error_for_valid_data(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20,
			brood_size = 3, brood_age = 10, labels = ['contact', 'feeding']
		)

		try:
			validate_recording_data(data)
		except Exception as e:
			self.fail(e)

	def test_value_error_for_labels_not_fit(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20,
			brood_size = 3, brood_age = 10, labels = ['contact', 'feeding']
		)
		data.labels.loc[-1, 'end'] = 125.0
		self.assertRaises(ValueError, lambda: validate_recording_data(data))

	def test_value_error_for_labels_invalid_timestamps(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20,
			brood_size = 3, brood_age = 10, labels = ['contact', 'feeding']
		)

		start = data.labels.start.iloc[0]
		end = data.labels.end.iloc[0]
		data.labels.loc[0, 'start'] = end + 1.0
		data.labels.loc[0, 'end'] = start

		self.assertRaises(ValueError, lambda: validate_recording_data(data))

	def test_value_error_for_unexpected_labels(self):
		labels = ['contact', 'feeding']
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20,
			brood_size = 3, brood_age = 10, labels = labels
		)
		data.labels.loc[0, 'label'] = 'unexpected label'
		self.assertRaises(ValueError, lambda: validate_recording_data(data, expected_labels = set(labels)))
