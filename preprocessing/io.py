from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf


@dataclass
class SnowfinchNestRecording:
	audio_data: np.ndarray
	audio_sample_rate: int
	labels: pd.DataFrame
	brood_age: int
	brood_size: int


def load_recording_data(data_path: str, recording_title: str) -> SnowfinchNestRecording:
	brood_age = number_from_recording_name(recording_title, label = 'BA', terminator = '_')
	brood_size = number_from_recording_name(recording_title, label = 'BS', terminator = '-')

	try:
		audio_data, sample_rate = sf.read(f'{data_path}/{recording_title}.flac')
		labels = pd.read_csv(
			f'{data_path}/{recording_title}.txt', sep = '\t',
			header = None, names = ['start', 'end', 'label']
		)
		return SnowfinchNestRecording(audio_data, sample_rate, labels, brood_age, brood_size)
	except sf.LibsndfileError:
		raise FileNotFoundError('Audio file not found')


def number_from_recording_name(recording_title: str, label: str, terminator: chr) -> int:
	try:
		start_idx = recording_title.index(label) + len(label)
		end_idx = recording_title.index(terminator, start_idx)
		return int(recording_title[start_idx:end_idx])
	except ValueError:
		raise ValueError(f'Invalid recording title format: failed to read {label} parameter')


def validate_recording_data(data: SnowfinchNestRecording, expected_labels: Optional[set[str]] = None):
	if not pd.Index(data.labels.start).is_monotonic_increasing:
		raise ValueError('Label start timestamps are not in ascending order')

	if not pd.Index(data.labels.end).is_monotonic_increasing:
		raise ValueError('Label end timestamps are not in ascending order')

	if any(data.labels.start >= data.labels.end):
		raise ValueError('Start timestamp of some labels is after their end timestamp')

	audio_length_sec = data.audio_data.shape[0] / data.audio_sample_rate
	if data.labels.start.iloc[0] < 0.0 or data.labels.end.iloc[-1] > audio_length_sec:
		raise ValueError('Labels timestamps do not fit in the audio')

	if expected_labels:
		for i in range(data.labels.shape[0]):
			if data.labels.label.iloc[i] not in expected_labels:
				raise ValueError(f'Unexpected label at position {i + 1}')
