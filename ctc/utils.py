import pickle
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import ctc

def load_files(
            emissions_path: str,
            transcript_path: str,
            labels_path: str,
            apply_softmax_on_emissions: bool = True) -> Tuple[torch.Tensor, str, List[str]]:
    with open(emissions_path, 'rb') as r:
        emissions = pickle.load(r)

    emissions = emissions.squeeze()
    if apply_softmax_on_emissions:
        emissions = F.softmax(emissions, dim=-1)

    labels = read_txt_file_lines(labels_path)
    transcript = read_txt_file_lines(transcript_path)
    transcript = transcript[0]

    assert len(labels) == emissions.shape[-1], \
        f"Last dimension of emission matrix of shape {emissions.shape} should be equal to the number of labels {len(labels)}.\
              Emissions matrix should be of shape (bsz, T, vocab_size)."
    return emissions, transcript, labels

def get_absolute_resources_file_path(relative_file_path: str):
    return os.path.join(os.path.dirname(os.path.abspath(ctc.__file__)), relative_file_path)

def read_txt_file_lines(text_fpath: str) -> List[str]:
    lines = []
    with open(text_fpath, 'r') as r:
        for line in r:
            lines.append(line.strip())
    return lines


def get_transcript_labels_and_IDs(
            transcript: str,
            label_to_id_dict: Dict[str, int],
            blank_token: str = '<s>',
            silence_token: str = '|') -> Tuple[List[str], List[int]]:
    transcript_labels = [blank_token]
    transcript_with_silence_token = transcript.replace(' ', silence_token) + '|'
    for c in transcript_with_silence_token:
        transcript_labels.extend([c, blank_token])
    transcript_ids = [label_to_id_dict[label] for label in transcript_labels]
    return transcript_labels, transcript_ids


def init_result_dict() -> Dict[str, List[float]]:
    result_dict = {
        'Forward Viterbi negative loglikelihood': [],
        'Backwrd Viterbi negative loglikelihood': [],
        'Torch CTC loss': [],
        'Implemented CTC Loss': [],
        'Gradient matrix norm': [],
        'Joint probability norm': [],
        'Emissions Gradient norm': [],
        'Case name': []
    }
    return result_dict


def update_result_dict(
            result_dict: dict,
            forward_likelihood: float,
            backward_likelihood: float,
            loss: float,
            gradient_matrix: np.ndarray,
            joint_probability_matrix: np.ndarray,
            torch_nll_loss: float,
            emissions_gradient_norm: float,
            case_name: str) -> Dict[str, float]:
    gradient_matrix_norm = np.linalg.norm(gradient_matrix)
    joint_probability_norm = np.linalg.norm(joint_probability_matrix)

    result_dict['Forward Viterbi negative loglikelihood'].append(forward_likelihood)
    result_dict['Backwrd Viterbi negative loglikelihood'].append(backward_likelihood)
    result_dict['Torch CTC loss'].append(torch_nll_loss)
    result_dict['Implemented CTC Loss'].append(loss)
    result_dict['Gradient matrix norm'].append(gradient_matrix_norm)
    result_dict['Joint probability norm'].append(joint_probability_norm)
    result_dict['Emissions Gradient norm'].append(emissions_gradient_norm)
    result_dict['Case name'].append(case_name)
    return result_dict


def get_result_dataframe_from_dict(result_dict: Dict[str, List[float]]) -> pd.DataFrame:
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df['Forward Viterbi negative loglikelihood'] = result_df['Forward Viterbi negative loglikelihood'].astype(float)
    result_df['Backwrd Viterbi negative loglikelihood'] = result_df['Backwrd Viterbi negative loglikelihood'].astype(float)
    result_df = result_df.set_index('Case name').sort_values(by='Forward Viterbi negative loglikelihood', ascending=True)
    return result_df


def set_emission_max_probs_to_one(emission_probabilities: torch.Tensor) -> torch.Tensor:
    eps = 0.00001
    capped_prob = 1 - eps
    capped_emission_probabilities = torch.zeros_like(emission_probabilities) + eps
    capped_emission_probabilities[:, torch.argmax(emission_probabilities, dim=-1)] = capped_prob
    return capped_emission_probabilities


def compute_torch_ctc_loss(
            acoustic_emission_probabilities: torch.Tensor,
            transcript_ids: List[int],
            blank_token_ID: int):
    nll_loss = F.ctc_loss(
            log_probs=torch.log(acoustic_emission_probabilities),
            targets=torch.tensor(transcript_ids, dtype=torch.long),
            input_lengths=[len(acoustic_emission_probabilities)],
            target_lengths=[len(transcript_ids)],
            blank=blank_token_ID,
            reduction='none')
    return nll_loss.item()
