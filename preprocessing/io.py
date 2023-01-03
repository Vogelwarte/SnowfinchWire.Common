from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import soundfile as sf
import csv
from typing import Callable
from pathlib import Path


@dataclass
class InputRecord:
	start: float
	end: float
	label: str


@dataclass
class SnowfinchNestRecording:
	title: str
	audio_data: np.ndarray
	audio_sample_rate: int
	labels: pd.DataFrame
	brood_age: int
	brood_size: int

	@property
	def audio_len_sec(self):
		return len(self.audio_data) * self.audio_sample_rate


def read_audacity_labels(data_path: Union[str, Path]) -> list[InputRecord]:
	result = []
	with open(data_path) as f:
		cf = csv.DictReader(f, fieldnames = ['start', 'end', 'label'], delimiter = '\t')
		for row in cf:
			result.append(InputRecord(float(row['start']), float(row['end']), row['label']))

	return result


def load_recording_data(
		data_path: Path, recording_title: Optional[str] = None,
		label_reader: Callable[[Union[str, Path]], list[InputRecord]] = read_audacity_labels
) -> SnowfinchNestRecording:
	if recording_title is None:
		recording_title = data_path.stem
		data_path = data_path.parent

	brood_age = number_from_recording_name(recording_title, label = 'BA', terminator = '_')
	brood_size = number_from_recording_name(recording_title, label = 'BS', terminator = '-')

	try:
		audio_data, sample_rate = sf.read(f'{data_path}/{recording_title}.flac')
		labels_file = next(Path(f'{data_path}').glob(f'{recording_title}*.txt'))
		labels_list = label_reader(labels_file)
		labels = pd.DataFrame(labels_list).convert_dtypes()
		return SnowfinchNestRecording(recording_title, audio_data, sample_rate, labels, brood_age, brood_size)
	except sf.LibsndfileError:
		raise FileNotFoundError('Audio file not found')
	except StopIteration:
		raise FileNotFoundError('Labels file not found')


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
