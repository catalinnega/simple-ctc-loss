from typing import List, Tuple, Union
import numpy as np
import torch
from torch.autograd import Function
from ctc.ctc_loss_processor import CTCLossProcessor

class CTCLoss(Function):
    @staticmethod
    def forward(ctx, emissions_matrix: torch.Tensor, transcript_IDs: torch.Tensor) -> float:
        forward_matrix, backward_matrix, forward_likelihood, _ = compute_viterbi_runs(emissions_matrix, transcript_IDs)
        forward_matrix, backward_matrix = torch.tensor(forward_matrix, dtype=torch.float16), torch.tensor(backward_matrix, dtype=torch.float16)
        ctx.save_for_backward(emissions_matrix, forward_matrix, backward_matrix, transcript_IDs)
        return -forward_likelihood

    @staticmethod
    def backward(ctx, grad_output):
        emissions_matrix, forward_matrix, backward_matrix, transcript_IDs = ctx.saved_tensors
        gradient_matrix, _ = compute_gradient_and_joint_probability_matrix(
                                                                    emissions_matrix=emissions_matrix,
                                                                    forward_matrix=forward_matrix,
                                                                    backward_matrix=backward_matrix,
                                                                    transcript_IDs=transcript_IDs,
                                                                    n_ctc_labels=forward_matrix.shape[0],
                                                                    blank_token_ID=0
                                                                )
        return gradient_matrix, grad_output*gradient_matrix

def compute_viterbi_runs(
        emissions_matrix: Union[torch.Tensor, np.array],
        transcript_IDs: List[int],
        blank_token_ID: int = 0
        )-> Tuple[np.array, np.array, float, float]:
    ctc_loss = CTCLossProcessor(blank_token_ID=blank_token_ID)
    return ctc_loss.compute_viterbi_runs(emissions_matrix, transcript_IDs)

def compute_gradient_and_joint_probability_matrix(
        emissions_matrix: Union[torch.Tensor, np.array],
        forward_matrix: np.array,
        backward_matrix: np.array,
        transcript_IDs: List[int],
        n_ctc_labels: int,
        blank_token_ID: int) -> Tuple[Union[torch.Tensor, np.array], np.array]:
    ctc_loss = CTCLossProcessor(blank_token_ID=blank_token_ID)
    return ctc_loss.compute_gradient_and_joint_probability_matrix(
                                                                emissions_matrix=emissions_matrix,
                                                                forward_matrix=forward_matrix,
                                                                backward_matrix=backward_matrix,
                                                                transcript_IDs=transcript_IDs,
                                                                n_ctc_labels=n_ctc_labels,
                                                                blank_token_ID=blank_token_ID
                                                                )

