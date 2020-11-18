from ast import literal_eval
import os
import json
import configparser
import hpbandster.core.result as hpres

def print_dict(dictionary):
	# Print contents of dict in json like format
	print(json.dumps(dictionary, indent=4))


class config_reader(object):
	def __init__(self, 	model_path: str):
		self.model_path = model_path

		self.labels_config_path = os.path.join(self.model_path, 'config','labels_config_file.ini')
		self.data_config_path = os.path.join(self.model_path, 'config','data_config_file.ini')
		self.labels_parameters = None
		self.data_parameters = None

	def get_model_path(self):
		return self.model_path
	
	def print_labels_parameters(self):
		print_dict(self.get_labels_parameters())

	def print_data_parameters(self):
		print_dict(self.get_data_parameters())

	def overwrite_data_parameters(self,alternative_data_parameters):
		print('Using alternative data parameters.')
		self.data_parameters = alternative_data_parameters

	def get_labels_parameters(self):
		if self.labels_parameters is None:
			self.read_label_config_file()
		return self.labels_parameters 


	def get_data_parameters(self):
		if self.data_parameters is None:
			self.read_data_config_file()
		return self.data_parameters 


	def read_label_config_file(self):
		if not os.path.exists(self.labels_config_path):
			raise AttributeError('Label config file does not exist.')
		config = configparser.ConfigParser()
		config.read(self.labels_config_path)

		labels_version = config['labels']['labels_version']
		if labels_version == 'threshold':
			self.labels_parameters = {
				'labels_version': labels_version,
				'distance': int(config[labels_version]['distance']),
				'buy_threshold': float(config[labels_version]['buy_threshold']),
				'sell_threshold': float(config[labels_version]['sell_threshold']),
				'n_limits': int(config[labels_version]['n_limits']),
				'window': int(config[labels_version]['window']),
				'coefpower': int(config[labels_version]['coefpower']),
				'boundary': float(config[labels_version]['boundary'])
			}
		else:
			if labels_version == 'up_down':
				self.labels_parameters = {
					'labels_version': labels_version,
					'up_limit': float(config[labels_version]['up_limit']),
					'down_limit': float(config[labels_version]['down_limit']),
					'coefpower': float(config[labels_version]['coefpower']),
					'window': int(config[labels_version]['window']),
					'boundary': float(config[labels_version]['boundary']),
					'n_limits': int(config[labels_version]['n_limits'])
				}
			else:
				raise AttributeError('unknown label version')

	def read_data_config_file(self):
		if not os.path.exists(self.data_config_path):
			raise AttributeError('Data config file does not exist.')

		config = configparser.ConfigParser()
		config.read(self.data_config_path)
		
		data_version = config['data']['data_version']
		if data_version == 'single_exchange':
			self.data_parameters = {
				'market_train': config[data_version+'_train']['market'],
				'exchange_train': config[data_version+'_train']['exchange'],
				'date_start_train': config[data_version+'_train']['date_start'],
				'date_end_train': config[data_version+'_train']['date_end'],

				'market_test': config[data_version+'_test']['market'],
				'exchange_test': config[data_version+'_test']['exchange'],
				'date_start_test': config[data_version+'_test']['date_start'],
				'date_end_test': config[data_version+'_test']['date_end']
			}
		else:
			raise AttributeError('unknown data version')