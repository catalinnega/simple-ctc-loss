from typing import List, Tuple, Generator, Union, Dict
import numpy as np
import torch

class CTCLossProcessor():
    def __init__(self, blank_token_ID: int = 0):
        self.blank_token_ID = blank_token_ID

    def compute_viterbi_runs(
            self,
            emissions_matrix: Union[torch.Tensor, np.array],
            transcript_IDs: List[int])-> Tuple[np.array, np.array, float, float]:
        """
        emissions_matrix: Matrix tensor representing the label probabilities per timeframes of shape (n_timeframes, vocab_size);
        transcript_IDs: Transcript IDs, with blank token at start, between and after label IDs;
        blank_token_ID: ID of blank token;

        returns tuple of:
            - forward matrix
            - backward matrix
            - negative log likelihood of computed forward Viterbi likelihood
            - negative log likelihood of computed backward Viterbi likelihood
        """

        n_timesteps = len(emissions_matrix)
        n_ctc_labels = 2 * len(transcript_IDs) + 1  # blank token inserted at start, between and after
        forward_matrix, forward_likelihood, backward_matrix, backward_likelihood = \
            self._initialize_viterbi_matrices_and_likelihoods(
                                        emissions_matrix=emissions_matrix,
                                        blank_token_ID=self.blank_token_ID,
                                        n_timesteps=n_timesteps,
                                        n_ctc_labels=n_ctc_labels,
                                        transcript_IDs=transcript_IDs)

        forward_matrix, forward_likelihood = self._run_viterbi_pass(
                                        run_type='forward',
                                        viterbi_matrix=forward_matrix,
                                        likelihood_estimate=forward_likelihood,
                                        emissions_matrix=emissions_matrix,
                                        blank_token_ID=self.blank_token_ID,
                                        transcript_IDs=transcript_IDs,
                                        n_timesteps=n_timesteps,
                                        n_ctc_labels=n_ctc_labels)

        backward_matrix, backward_likelihood = self._run_viterbi_pass(
                                        run_type='backward',
                                        viterbi_matrix=backward_matrix,
                                        likelihood_estimate=backward_likelihood,
                                        emissions_matrix=emissions_matrix,
                                        blank_token_ID=self.blank_token_ID,
                                        transcript_IDs=transcript_IDs,
                                        n_timesteps=n_timesteps,
                                        n_ctc_labels=n_ctc_labels)
        return forward_matrix, backward_matrix, forward_likelihood, backward_likelihood

    def compute_gradient_and_joint_probability_matrix(
                self,
                emissions_matrix: Union[torch.Tensor, np.array],
                forward_matrix: torch.Tensor,
                backward_matrix: torch.Tensor,
                transcript_IDs: List[int],
                n_ctc_labels: int,
                blank_token_ID: int) -> Tuple[Union[torch.Tensor, np.array], np.array]:
        gradient_matrix = torch.zeros_like(emissions_matrix)
        joint_probability_matrix = forward_matrix * backward_matrix
        for label_idx in range(n_ctc_labels):
            if label_idx % 2 == 0:
                # case: blank token
                local_idx = blank_token_ID

            else:
                # case: non-blank token
                non_blank_label_IDs_idx = int((label_idx - 1) / 2)
                non_blank_label_idx = transcript_IDs[non_blank_label_IDs_idx]
                local_idx = non_blank_label_idx
            gradient_matrix[:, local_idx] += joint_probability_matrix[label_idx, :]
            joint_probability_matrix[label_idx, :] /= emissions_matrix[:, local_idx]

        sum_joint_probabilities = torch.sum(joint_probability_matrix, dim=0)
        normed_emissions_matrix = emissions_matrix.transpose(0, 1) * sum_joint_probabilities
        normed_emissions_matrix = normed_emissions_matrix.transpose(0, 1)
        gradient_matrix = emissions_matrix - gradient_matrix / normed_emissions_matrix
        return gradient_matrix, joint_probability_matrix

    def viterbi_decoding(
                self,
                acoustic_emissions: torch.Tensor,
                id_to_label_dict: Dict[int, str]) -> str:
        '''
        Computes greedy decoding (arg max) on acoustic emission probability matrix.

        acoustic_emissions shape: (n_timesteps, vocab_size)
        '''
        token_IDs = acoustic_emissions.argmax(dim=-1).unique_consecutive()
        decoded_text = [id_to_label_dict[t] for t in token_IDs.numpy()]
        return decoded_text

    def _initialize_viterbi_matrices_and_likelihoods(
                self,
                emissions_matrix: Union[torch.Tensor, np.array],
                blank_token_ID: int,
                n_timesteps: int,
                n_ctc_labels: int,
                transcript_IDs: List[int]) -> Tuple[np.array, int, np.array, int]:
        forward_matrix, forward_likelihood = self._initialize_viterbi_matrix_and_likelihood(
                    n_ctc_labels=n_ctc_labels,
                    n_timesteps=n_timesteps,
                    blank_prob=emissions_matrix[0, blank_token_ID],
                    label_prob=emissions_matrix[0, transcript_IDs[0]],
                    type='forward')
        backward_matrix, backward_likelihood = self._initialize_viterbi_matrix_and_likelihood(
                    n_ctc_labels=n_ctc_labels,
                    n_timesteps=n_timesteps,
                    blank_prob=emissions_matrix[-1, blank_token_ID],
                    label_prob=emissions_matrix[-1, transcript_IDs[0]],
                    type='backward')
        return forward_matrix, forward_likelihood, backward_matrix, backward_likelihood

    def _initialize_viterbi_matrix_and_likelihood(
            self,
            n_ctc_labels: int,
            n_timesteps: int,
            blank_prob: float,
            label_prob: float,
            type: str) -> Tuple[np.array, float]:
        '''
        https://www.cs.toronto.edu/~graves/icml_2006.pdf
            section 4, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
        '''
        viterbi_matrix = np.zeros((n_ctc_labels, n_timesteps))
        prob_sum = blank_prob + label_prob
        if type == 'forward':
            viterbi_matrix[0, 0] = blank_prob / prob_sum
            viterbi_matrix[1, 0] = label_prob / prob_sum
        elif type == 'backward':
            viterbi_matrix[-1, -1] = blank_prob / prob_sum
            viterbi_matrix[-2, -1] = label_prob / prob_sum
        else:
            print(f"Error. Unknown Viterbi matrix type: {type}")
        likelihood = np.log(prob_sum)
        return viterbi_matrix, likelihood

    def _run_viterbi_pass(
                self,
                viterbi_matrix: np.array,
                likelihood_estimate: float,
                emissions_matrix: Union[torch.Tensor, np.array],
                blank_token_ID: int,
                transcript_IDs: List[int],
                n_timesteps: int,
                n_ctc_labels: int,
                run_type: str) -> Tuple[np.array, float]:
        index_direction_scaling = -1 if run_type == 'forward' else 1
        timesteps_range = self._get_timesteps_range(run_type, n_timesteps)
        for timestep in timesteps_range:
            start_timestep_index, end_timestep_index = self._get_start_and_end_index_viterbi_timesteps(
                                                                current_index=timestep,
                                                                n_timesteps=n_timesteps,
                                                                n_ctc_labels=n_ctc_labels)

            label_range = self._get_labels_range(run_type, start_timestep_index, end_timestep_index, n_ctc_labels)
            for label_index in label_range:
                non_blank_label_index = int((label_index - 1) / 2)

                emission_label_prob = emissions_matrix[timestep, transcript_IDs[non_blank_label_index]]
                emission_blank_prob = emissions_matrix[timestep, blank_token_ID]

                current_label_next_prob = viterbi_matrix[label_index, timestep + 1 * index_direction_scaling]
                next_label_next_prob, second_next_label_next_prob = self._get_next_label_probabilities(
                                                                                run_type=run_type,
                                                                                viterbi_matrix=viterbi_matrix,
                                                                                label_index=label_index,
                                                                                timestep=timestep,
                                                                                transcript_IDs=transcript_IDs,
                                                                                blank_token_ID=blank_token_ID,
                                                                                n_ctc_labels=n_ctc_labels,
                                                                                non_blank_label_index=non_blank_label_index,
                                                                                index_scale=index_direction_scaling
                                                                            )
                if label_index % 2 == 0:
                    # case: blank token
                    viterbi_matrix[label_index, timestep] = current_label_next_prob * emission_blank_prob
                    if self._viterbi_blank_token_condition(run_type, label_index, blank_token_ID, n_ctc_labels):
                        viterbi_matrix[label_index, timestep] += next_label_next_prob * emission_blank_prob
                elif self._viterbi_same_token_condition(run_type, label_index, transcript_IDs, non_blank_label_index, n_ctc_labels):
                    # case: two consecutive labels
                    viterbi_matrix[label_index, timestep] = (current_label_next_prob + next_label_next_prob) * emission_label_prob
                else:
                    # case: distinct consecutive labels
                    viterbi_matrix[label_index, timestep] = \
                        (current_label_next_prob + next_label_next_prob + second_next_label_next_prob) * emission_label_prob
            likelihood_sum, viterbi_matrix = self._normalize_viterbi_matrix(
                viterbi_matrix=viterbi_matrix,
                start_timestep_index=start_timestep_index,
                end_timestep_index=end_timestep_index,
                current_timestep_index=timestep
            )
            likelihood_estimate += np.log(likelihood_sum)
        return viterbi_matrix, likelihood_estimate

    def _get_timesteps_range(self, run_type: str, n_timesteps: int) -> Generator[int, None, None]:
        if run_type == 'forward':
            timesteps_range = range(1, n_timesteps)
        else:
            timesteps_range = range(n_timesteps - 2, -1, -1)
        return timesteps_range

    def _get_labels_range(
                    self,
                    run_type: str,
                    start_timestep_index: int,
                    end_timestep_index: int,
                    n_ctc_labels: int) -> Generator[int, None, None]:
        if run_type == 'forward':
            label_range = range(start_timestep_index, n_ctc_labels)
        else:
            label_range = range(end_timestep_index - 1, -1, -1)
        return label_range

    def _get_start_and_end_index_viterbi_timesteps(
                self,
                current_index: int,
                n_timesteps: int,
                n_ctc_labels: int) -> Tuple[int, int]:
        '''
        https://www.cs.toronto.edu/~graves/icml_2006.pdf
        "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
            ".. these variables correspond to states for which there are not enough time-steps left to complete the sequence"
        '''
        start_index = n_ctc_labels - 2 * (n_timesteps - current_index)
        start_index = start_index if start_index > 0 else 0

        end_index = 2 * current_index + 2
        end_index = end_index if end_index < n_ctc_labels else n_ctc_labels
        return start_index, end_index

    def _viterbi_blank_token_condition(
                self,
                run_type: str,
                label_index: int,
                blank_token_ID: int,
                n_ctc_labels: int) -> bool:
        if run_type == 'forward':
            return label_index != blank_token_ID
        else:
            return label_index != n_ctc_labels - 1

    def _viterbi_same_token_condition(
                self,
                run_type: str,
                label_index: int,
                transcript_IDs: List[int],
                non_blank_label_index: int,
                n_ctc_labels: int) -> bool:
        if run_type == 'forward':
            return label_index == 1 or transcript_IDs[non_blank_label_index] == transcript_IDs[non_blank_label_index - 1]
        else:
            return label_index == n_ctc_labels - 2 or \
                transcript_IDs[non_blank_label_index] == transcript_IDs[non_blank_label_index + 1]

    def _normalize_viterbi_matrix(
                self,
                viterbi_matrix: np.array,
                start_timestep_index: int,
                end_timestep_index: int,
                current_timestep_index) -> Tuple[float, np.array]:
        likelihood_sum = np.sum(viterbi_matrix[start_timestep_index:end_timestep_index, current_timestep_index])
        viterbi_matrix[start_timestep_index:end_timestep_index, current_timestep_index] /= likelihood_sum
        return likelihood_sum, viterbi_matrix

    def _get_next_label_probabilities(
                self,
                run_type: str,
                viterbi_matrix: np.array,
                label_index: int,
                timestep: int,
                transcript_IDs: List[int],
                blank_token_ID: int,
                n_ctc_labels: int,
                non_blank_label_index: int,
                index_scale: int):
        if run_type == 'backward':
            # avoid index overflow cases
            next_label_next_prob, second_next_label_next_prob = None, None
            if label_index % 2 == 0:
                if self._viterbi_blank_token_condition(run_type, label_index, blank_token_ID, n_ctc_labels):
                    next_label_next_prob = viterbi_matrix[label_index + 1 * index_scale, timestep + 1 * index_scale]
            elif self._viterbi_same_token_condition(run_type, label_index, transcript_IDs, non_blank_label_index, n_ctc_labels):
                next_label_next_prob = viterbi_matrix[label_index + 1 * index_scale, timestep + 1 * index_scale]
            else:
                next_label_next_prob = viterbi_matrix[label_index + 1 * index_scale, timestep + 1 * index_scale]
                second_next_label_next_prob = viterbi_matrix[label_index + 2 * index_scale, timestep + 1 * index_scale]
        else:
            next_label_next_prob = viterbi_matrix[label_index + 1 * index_scale, timestep + 1 * index_scale]
            second_next_label_next_prob = viterbi_matrix[label_index + 2 * index_scale, timestep + 1 * index_scale]
        return next_label_next_prob, second_next_label_next_prob