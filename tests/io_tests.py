import unittest

import numpy as np
import pandas as pd

from common.io import SnowfinchNestRecording, validate_recording_data


def generate_audio(sample_rate: int, length_sec: int) -> np.ndarray:
	return np.random.random(sample_rate * length_sec) * 2.0 - 1.0


def generate_labels(start: float, end: float, count: int) -> pd.DataFrame:
	max_label_length_sec = (end - start) / count

	label_starts = np.linspace(start, end - max_label_length_sec, num = count)
	label_ends = label_starts + np.random.random(count) * max_label_length_sec
	label_options = ['contact', 'feeding']
	label_indices = np.random.randint(0, len(label_options), count)

	return pd.DataFrame(
		data = {
			'start': label_starts,
			'end': label_ends,
			'label': np.array(label_options)[label_indices]
		}
	)


def generate_nest_recoring(
		sample_rate: int, length_sec: int, label_count: int, brood_age: int, brood_size: int
) -> SnowfinchNestRecording:
	audio = generate_audio(sample_rate, length_sec)
	labels = generate_labels(0.0, length_sec, label_count)
	return SnowfinchNestRecording(audio, sample_rate, labels, brood_age, brood_size)


class TestValidateRecordingData(unittest.TestCase):
	def test_no_error_for_valid_data(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20, brood_size = 3, brood_age = 10
		)

		try:
			validate_recording_data(data)
		except Exception as e:
			self.fail(e)

	def test_value_error_for_labels_not_fit(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20, brood_size = 3, brood_age = 10
		)
		data.labels.end.iloc[-1] = 125.0
		self.assertRaises(ValueError, lambda: validate_recording_data(data))

	def test_value_error_for_labels_invalid_timestamps(self):
		data = generate_nest_recoring(
			sample_rate = 48000, length_sec = 120, label_count = 20, brood_size = 3, brood_age = 10
		)

		start = data.labels.start.iloc[0]
		end = data.labels.end.iloc[0]
		data.labels.start.iloc[0] = end + 1.0
		data.labels.end.iloc[0] = start

		self.assertRaises(ValueError, lambda: validate_recording_data(data))
