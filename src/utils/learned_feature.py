import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import trange
import itertools
from transform_input import transform_input, get_subranges
from networks import DNN
from torch.utils.data import Dataset, DataLoader


class LearnedFeature(object):
	"""
	Learned Feature Class contains the feature function as well as data and training functionality.
	----
	Initialize:
	nb_layers	number of hidden layers for the NN
	nb_units	number of hidden units per layer for the NN

	LF_dict		dict containing all settings for the learned feature, example see below

	LF_dict = {'bet_data':5, 'sin':False, 'cos':False, 'rpy':False, 'lowdim':False, 'norot':True,
           'noangles':True, '6D_laptop':False, '6D_human':False, '9D_coffee':False, 'EErot':False,
           'noxyz':False, 'subspace_heuristic': False}

	bet_data [int]		: Number of copies of each Start-Goal pair to be added to the data set.
	subspace_heuristic	: if True, the heuristic to select the angle, rotation matrices, or euclidean subspace is used

	sin, cos, noangles	: noangles means no angles in the input space, sin/cos and are transforms of the default 7 angles
	norot, EErot, rpy	: if norot True no rotation matrices are in the input space, if EErot only the rotation matrix
						of the endeffector and not of the other joints, rpy: the Roll/Pitch/Yaw representation is used
	lowdim				: only the endeffector xyz, and rotation matrix plus the angles & object xyz
	9D_coffee			: only the 9 entries of the Endeffector rotation matrix as input
	6D_laptop, 6D_human	: only the endeffector xyz plus the xyz of the laptop or the human is used as input space
	"""
	def __init__(self, nb_layers, nb_units, LF_dict):

		self.trace_list = []
		self.full_data_array = np.empty((0, 5), float)
		self.start_labels = []
		self.end_labels = []
		self.subspaces_list = get_subranges(LF_dict)
		self.max_labels = [1 for _ in range(len(self.subspaces_list))]
		self.min_labels = [0 for _ in range(len(self.subspaces_list))]
		self.LF_dict = LF_dict
		self.models = []

		# set default
		self.final_model = 0

		# ---- Initialize Function approximators for each subspace ---- #
		if self.LF_dict['subspace_heuristic']:
			for sub_range in self.subspaces_list:
				self.models.append(DNN(nb_layers, nb_units, sub_range[1] - sub_range[0]))
		else:
			self.models.append(DNN(nb_layers, nb_units, self.subspaces_list[-1][1]))

	def function(self, x, model=None, torchify=False, norm=False):
		"""
			Feature Function used for Training & in Trajopt after Feature Learning
			----
			Input:
			x			Nx97 tensor raw state input
			model		which model to use
			torchify	if the output should be a torch tensor
			norm		if the output should be normalized

			Output:
			y 			Nx1 output tensor
		"""

		if model is None: # then called from TrajOpt so return normalized final model
			model = self.final_model
			norm = True

		# Torchify the input
		x = self.input_torchify(x)

		# Transform the input
		x = transform_input(x, self.LF_dict)

		if self.LF_dict['subspace_heuristic']: # transform to the model specific subspace input
			sub_range = self.subspaces_list[model]
			x = x[:, sub_range[0]:sub_range[1]]
		y = self.models[model](x)

		if norm:
			y = (y - self.min_labels[model]) / (self.max_labels[model] - self.min_labels[model])
			if torchify:
				return y
			else:
				return np.array(y.detach())
		else:
			return y

	def input_torchify(self, x):
		"""
			Transforms numpy input to torch tensors.
		"""
		if not torch.is_tensor(x):
			x = torch.Tensor(x)
		if len(x.shape) == 1:
			x = torch.unsqueeze(x, axis=0)
		return x

	def add_data(self, feature_trace, start_label=0.0, end_label=1.0):
		"""
			Adding feature traces during data collection.
			----
			Input:
			feature_trace 	feature trace in angle space nx7 np.array
			start_label		feature value at start of the trace
			end_label		feature value at end of the trace
		"""
		self.trace_list.append(feature_trace)
		self.start_labels.append(start_label)
		self.end_labels.append(end_label)

	def get_train_test_arrays(self, train_idx, test_idx):
		"""
			Create the full data array of tuples and a train & test tuple set
			----
			Input:
			train_idx, test_idx		a list of the train/test indices of the trace_list

			Output: test & train data arrays of tuples
		"""
		full_data_array = np.empty((0, 5), float)
		ordered_list = train_idx + test_idx
		test_set_idx = None

		for idx in ordered_list:
			# check if already test set
			if idx == test_idx[0]:
				# log where that is so we can split the full array later
				test_set_idx = full_data_array.shape[0]
			data_tuples_to_append = []
			for combi in list(itertools.combinations(range(self.trace_list[idx].shape[0]), 2)):
				# Sample two points on that trajectory trace.
				idx_s0, idx_s1 = combi

				# Create label differentials if necessary.
				s0_delta = 0
				s1_delta = 0
				if idx_s0 == 0:
					s0_delta = -self.start_labels[idx]
				if idx_s1 == self.trace_list[idx].shape[0] - 1:
					s1_delta = 1. - self.end_labels[idx]

				data_tuples_to_append.append((
					self.trace_list[idx][idx_s0, :], self.trace_list[idx][idx_s1, :],
					idx_s0 < idx_s1, s0_delta, s1_delta))
			full_data_array = np.vstack((full_data_array, np.array(data_tuples_to_append)))

			# Add between FERL_traces tuples
			if ordered_list.index(idx) > 0:
				tuples_to_append = []
				for other_traj_idx in ordered_list[:ordered_list.index(idx)]:
					S_tuple = [(self.trace_list[other_traj_idx][0, :], self.trace_list[idx][0, :], 0.5,
								-self.start_labels[other_traj_idx], -self.start_labels[idx])] * self.LF_dict['bet_data']
					G_tuple = [(self.trace_list[other_traj_idx][-1, :], self.trace_list[idx][-1, :], 0.5,
								1 - self.end_labels[other_traj_idx], 1 - self.end_labels[idx])] * self.LF_dict['bet_data']
					tuples_to_append = tuples_to_append + S_tuple + G_tuple
				full_data_array = np.vstack((full_data_array, np.array(tuples_to_append)))

		# split in train & test tuples
		self.full_data_array = full_data_array
		train_data_array = full_data_array[:test_set_idx]
		test_data_array = full_data_array[test_set_idx:]

		return test_data_array, train_data_array

	def select_subspace(self, epochs, batch_size, learning_rate, weight_decay, s_g_weight):
		"""
			If heuristic: select the subspace model, otherwise: initialize optimizers & create data tuple array
			----
			Input:
			epochs, batch_size, learning_rate, weight_decay, s_g_weight

			Output: optimizer of the choosen final model
		"""
		if len(self.trace_list) == 1 and self.LF_dict['subspace_heuristic']:
			print("Subspace Heuristic needs at least two Feature Traces to work.")

		n_test = int(math.floor(len(self.trace_list) * 0.5))
		print("Select subspace training, testing on " + str(n_test) + " unseen Trajectory")
		# split trajectory list
		test_idx = np.random.choice(np.arange(len(self.trace_list)), size=n_test, replace=False)
		train_idx = np.setdiff1d(np.arange(len(self.trace_list)), test_idx)

		test_data_array, train_data_array = self.get_train_test_arrays(train_idx.tolist(), test_idx.tolist())

		train_dataset = FeatureLearningDataset(train_data_array)
		print("len train: ", train_data_array.shape[0])
		test_dataset = FeatureLearningDataset(test_data_array)
		print("len test: ", test_data_array.shape[0])
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

		optimizers = []
		for i in range(len(self.models)):
			# initialize optimizers
			optimizers.append(optim.Adam(self.models[i].parameters(), lr=learning_rate, weight_decay=weight_decay))

		# unnecessary if only one subspace
		if len(self.subspaces_list) ==1 or self.LF_dict['subspace_heuristic']:
			print("No subspace selection performed.")
			return optimizers[0]

		train_losses = [[] for _ in range(len(self.subspaces_list))]
		test_losses = [[] for _ in range(len(self.subspaces_list))]

		# check if any non-standard label is used
		norm_per_epoch = False
		if sum([l != 0 for l in self.start_labels]) > 0 or sum([l != 1 for l in self.end_labels]) > 0:
			norm_per_epoch = True

		with trange(epochs) as T:
			for t in T:
				# Description will be displayed on the left
				T.set_description('epoch %i' % i)

				# Needed if non-standard labeling is used
				if norm_per_epoch:
					for idx in range(len(self.models)):
						self.update_normalizer(idx)

				for i, model in enumerate(self.models):
					avg_in_loss = []
					for batch in train_loader:
						optimizers[i].zero_grad()
						loss = self.FERL_loss(batch, model_idx=i, s_g_weight=s_g_weight)
						loss.backward()
						optimizers[i].step()
						avg_in_loss.append(loss.item())
					# self.update_normalizer(i) # technically correct but costs a lot of compute

					train_losses[i].append(sum(avg_in_loss) / len(avg_in_loss))

				# calculate test loss
				for i, model in enumerate(self.models):
					avg_in_loss = []
					for batch in test_loader:
						loss = self.FERL_loss(batch, model_idx=i, s_g_weight=s_g_weight)
						avg_in_loss.append(loss.item())
					# log over training
					test_losses[i].append(sum(avg_in_loss) / len(avg_in_loss))

				T.set_postfix(test_loss=[loss[-1] for loss in test_losses])

		for idx in range(len(self.models)):
			self.update_normalizer(idx)

		# select final model; Take lowest last test loss
		final_test_losses = [loss[-1] for loss in test_losses]
		val, last_loss_idx = min((val, idx) for (idx, val) in enumerate(final_test_losses))

		print("Model of subspace " + str(last_loss_idx) + "selected as final model")
		self.final_model = last_loss_idx

		return optimizers[last_loss_idx]

	def update_normalizer(self, model_idx):
		"""
			Update the max & min labels used for normalization based on all trace tuples in the dataset.
			----
			Input:
			model_idx	index of the model to normalize
		"""
		s_0s_array = np.array([tup[0] for tup in self.full_data_array]).squeeze()
		s_1s_array = np.array([tup[1] for tup in self.full_data_array]).squeeze()
		s0_logits = self.function(s_0s_array, model=model_idx).view(-1).detach()
		s1_logits = self.function(s_1s_array, model=model_idx).view(-1).detach()
		all_logits = np.vstack((s0_logits, s1_logits))
		self.max_labels[model_idx] = np.amax(all_logits)
		self.min_labels[model_idx] = np.amin(all_logits)

	def FERL_loss(self, batch, model_idx, s_g_weight):
		"""
			Calculate the FERL Loss for a model index, start & goal tuple weights, over a batch of tuples.
			----
			Input:
			batch		dict with the batch data of tuples (s1, s2, l1, l2, label)
			model_idx	index of the model to calculate the loss for
			s_g_weight	numeric value how strong tuples of the start & end state of traces are weighted in the loss

			output: 	scalar loss
		"""
		# arrays of the states
		s_1s_array = batch['s1']
		s_2s_array = batch['s2']

		# arrays of the start & end labels
		delta_1s_array = batch['l1']
		delta_2s_array = batch['l2']

		# label for classifiers
		labels = batch['label']

		weights = torch.ones(labels.shape)
		weights = weights + (labels == 0.5)*torch.full(labels.shape, s_g_weight)

		s1_adds = (delta_1s_array * (self.max_labels[model_idx] - self.min_labels[model_idx])).reshape(-1,1)
		s2_adds = (delta_2s_array * (self.max_labels[model_idx] - self.min_labels[model_idx])).reshape(-1, 1)

		# calculate test_loss  (with additive thing)
		s1_logits = self.function(s_1s_array, model=model_idx) + s1_adds
		s2_logits = self.function(s_2s_array, model=model_idx) + s2_adds

		# final loss
		loss = nn.BCEWithLogitsLoss(weight=weights)((s2_logits - s1_logits).view(-1), labels)
		return loss

	def train(self, epochs=100, batch_size=32, learning_rate=1e-3, weight_decay=0.001, s_g_weight=10.):
		"""
			Train the Feature function with the current data. Settings self-explanatory.
			----
			Input:
			epochs, batch_size, learning_rate, weight_decay, s_g_weight

			Output: train_losses as list
		"""
		# Heuristic to select subspace by training for 10 epochs a NN on each of them
		final_mod_optimizer = self.select_subspace(10, batch_size, learning_rate, weight_decay, s_g_weight)
		# Train on full dataset
		print("Train subspace model " + str(self.final_model) + " on all " + str(
			len(self.trace_list)) + " Trajectories")
		train_dataset = FeatureLearningDataset(self.full_data_array)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

		# check if any non-standard label is used
		norm_per_epoch = False
		if sum([l != 0 for l in self.start_labels]) > 0 or sum([l != 1 for l in self.end_labels]) > 0:
			norm_per_epoch = True

		train_losses = []
		with trange(epochs) as T:
			for t in T:
				# Description will be displayed on the left
				T.set_description('epoch %i' % t)

				# update normalizer labels
				if norm_per_epoch:
					self.update_normalizer(self.final_model)

				avg_in_loss = []
				for batch in train_loader:
					final_mod_optimizer.zero_grad()
					loss = self.FERL_loss(batch, model_idx=self.final_model, s_g_weight=s_g_weight)
					loss.backward()
					final_mod_optimizer.step()
					avg_in_loss.append(loss.item())

					train_losses.append(sum(avg_in_loss) / len(avg_in_loss))

				T.set_postfix(train_loss=train_losses[-1])

		self.update_normalizer(self.final_model)

		print("Final model trained!")
		return train_losses


class FeatureLearningDataset(Dataset):
	"""Feature Learning dataset of Tuples."""

	def __init__(self, array_of_tuples):
		self.array_of_tuples = array_of_tuples

	def __len__(self):
		return self.array_of_tuples.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample = {'s1': self.array_of_tuples[idx][0].astype(np.float32),
				  's2': self.array_of_tuples[idx][1].astype(np.float32),
				  'label': np.array(self.array_of_tuples[idx][2]).astype(np.float32),
				  'l1': np.array(self.array_of_tuples[idx][3]).astype(np.float32),
				  'l2': np.array(self.array_of_tuples[idx][4]).astype(np.float32)
				  }

		return sample
