import pandas as pd
import re
from tqdm import tqdm


def get_columns(config):
	columns = {}
	for col in config['columns']:
		p = col['position']
		columns[p] = col['mappedTo']
	columns = sorted(columns.items(), key=lambda x: x[0])
	return zip(*columns)


def split_lines(filename, separator, mappings, use_cols):
	"""
	Reads gaze file and splits lines two separate groups:
	Main lines: The usual gaze data that is composed of several entries such as timestamp, x and y.
	Special lines: This can be anything that does not conform to the format of the main data, such as special events (fixations, saccades, etc.)
	For example Eyelink ascii files encode gaze events such as fixations and saccades as special lines.
	"""
	special_lines = {}
	main_lines = []

	with open(filename, 'r') as f:
		for line in tqdm(f.readlines(), unit='line', desc='Preprocess lines', disable=True):
			line = line.strip('\n')
			if len(line) == 0:
				continue
			for m in mappings:
				mappedTo = m['mappedTo']
				if re.match(m['pattern'], line) is not None:
					if mappedTo not in special_lines:
						special_lines[mappedTo] = []
					special_lines[mappedTo].append(line)
					break
			else:
				entries = line.split(separator)
				entries = [entries[idx] for idx in use_cols]
				main_lines.append(entries)
	return main_lines, special_lines


def parse_gaze(filename, config):
	positions, column_names = get_columns(config)

	if 'special_lines' in config:
		mappings = config['special_lines']
		mappings.append({'pattern': f'^{re.escape(config["comment"])}', 'mappedTo': ''})
		main_lines, special_lines = split_lines(filename, use_cols=positions, separator=config['separator'], mappings=mappings)
		df = pd.DataFrame(data=main_lines, columns=column_names)
	else:
		special_lines = []
		df = pd.read_csv(filename, names=column_names, header=None, usecols=positions, sep=config['separator'], comment=config['comment'])

	df = df[config['skip_header_lines']:]

	for c in column_names:
		if c != 'EVENT':
			df[c] = df[c].astype(float)

	if 'column_transform' in config:
		for transform in config['column_transform']:
			col = transform['column']

			coeff_a = transform['linear-coeff']['a']
			coeff_b = transform['linear-coeff']['b']

			assert type(coeff_a) in (int, float), "Coefficient a in linear transform must be numeric!"
			assert type(coeff_b) in (int, float), "Coefficient a in linear transform must be numeric!"

			df[col] = df[col].apply(lambda x: (x*coeff_a+coeff_b))
	return df, special_lines


if __name__=='__main__':
	import argparse
	import json

	parser = argparse.ArgumentParser()
	parser.add_argument('--gaze_path', type=str, required=True)
	parser.add_argument('--config_path', type=str, required=True)
	args = parser.parse_args()

	with open(args.config_path, 'r') as f:
		config = json.load(f)
		gaze, special_lines = parse_gaze(args.gaze_path, config)
		print(gaze)
		print(special_lines)