import argparse

import torch

from ctc_loss_processor import CTCLossProcessor
from ctc_loss import CTCLoss
from utils import load_files,\
                init_result_dict,\
                update_result_dict,\
                get_transcript_labels_and_IDs,\
                set_emission_max_probs_to_one,\
                get_result_dataframe_from_dict,\
                get_absolute_resources_file_path,\
                compute_torch_ctc_loss


def main(emissions_path: str, transcript_path: str, labels_path: str, blank_token_ID: int = 0):
    acoustic_emission_probabilities, transcript, labels = load_files(
                                                emissions_path,
                                                transcript_path,
                                                labels_path,
                                                apply_softmax_on_emissions=True)

    label_to_id_dict = {label: idx for idx, label in enumerate(labels)}
    id_to_label_dict = {idx: label for label, idx in label_to_id_dict.items()}

    transcript_labels, transcript_ids = get_transcript_labels_and_IDs(
                                                transcript,
                                                label_to_id_dict,
                                                blank_token='<s>',
                                                silence_token='|')
    ctc_loss_processor = CTCLossProcessor(blank_token_ID=blank_token_ID)
    ctc_loss = CTCLoss.apply

    test_transcript_ids_cases = [
        ['Transcript IDs', transcript_ids],
        ['Transcript IDs with max probabilities == 1', transcript_ids],
        ['No blank separators', [label_to_id_dict[c] for c in transcript.replace(' ', '|')]],
        ['Half of IDs are wrong', [abs(label_id - 1) if label_id % 2 == 0 else label_id for label_id in transcript_ids]],
        ['All IDs are the same', [label_to_id_dict['X'] for _ in transcript_ids]]
    ]

    result_dict = init_result_dict()
    for case_name, transcript_ids in test_transcript_ids_cases:
        if case_name == 'Transcript IDs with max probabilities == 1':
            case_emission_probabilities = set_emission_max_probs_to_one(acoustic_emission_probabilities).clone()
        else:
            case_emission_probabilities = acoustic_emission_probabilities.clone()

        forward_matrix, backward_matrix, forward_likelihood, backward_likelihood = ctc_loss_processor.compute_viterbi_runs(
                                                                        case_emission_probabilities,
                                                                        transcript_ids)

        gradient_matrix, joint_probability_matrix = ctc_loss_processor.compute_gradient_and_joint_probability_matrix(
                                                                    emissions_matrix=case_emission_probabilities,
                                                                    forward_matrix=torch.tensor(forward_matrix),
                                                                    backward_matrix=torch.tensor(backward_matrix),
                                                                    transcript_IDs=transcript_ids,
                                                                    n_ctc_labels=forward_matrix.shape[0],
                                                                    blank_token_ID=blank_token_ID
                                                                )
        case_emission_probabilities = case_emission_probabilities.requires_grad_(True)
        loss = ctc_loss(case_emission_probabilities, torch.tensor(transcript_ids, dtype=torch.long))
        loss.backward()

        emissions_gradient_norm = torch.norm(case_emission_probabilities.grad)

        torch_nll_loss = compute_torch_ctc_loss(
                                case_emission_probabilities,
                                transcript_ids,
                                blank_token_ID=blank_token_ID)

        result_dict = update_result_dict(
                                result_dict,
                                forward_likelihood,
                                backward_likelihood,
                                loss.item(),
                                gradient_matrix,
                                joint_probability_matrix,
                                torch_nll_loss,
                                emissions_gradient_norm,
                                case_name)
    viterbi_labels = ctc_loss_processor.viterbi_decoding(acoustic_emission_probabilities, id_to_label_dict)

    result_df = get_result_dataframe_from_dict(result_dict)
    print("\n----------\nResults summary:\n")
    print(f"Acoustic emissions matrix shape:\n\t{acoustic_emission_probabilities.shape}\n")
    print(f"Label to ID dictionary:\n\t{label_to_id_dict}\n")
    print(f"Transcript text:\n\t{transcript}\n")
    print(f"Transcript labels:\n\t{transcript_labels}\nViterbi (argmax) decoding labels:\n\t{viterbi_labels}")
    print(result_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
                '--emissions_path',
                default=get_absolute_resources_file_path('../resources/ctc/emissions.pkl'),
                action='store',
                type=str,
                required=False,
                help='Pickle file containing acoustic model emissions matrix of shape (bsz, T, vocab_size).')
    parser.add_argument(
                '--transcript_path',
                default=get_absolute_resources_file_path('../resources/ctc/emissions_transcript.txt'),
                action='store',
                type=str,
                required=False,
                help='Text file containing the correct transcript sequence associated with the emission matrix.')
    parser.add_argument(
                '--labels_path',
                default=get_absolute_resources_file_path('../resources/ctc/labels.txt'),
                action='store',
                type=str,
                required=False,
                help='Text file containing acoustic model labels, each on one line.')
    args = parser.parse_args()
    main(args.emissions_path, args.transcript_path, args.labels_path)
