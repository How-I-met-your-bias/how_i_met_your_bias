"""
Nathan Roos
"""

import torch


def create_balanced_dataset(
    train_dataset: torch.utils.data.Dataset,
    rho: float = None,
    align_count: int = None,
    conflict_count: int = None,
):
    """
    Create a balanced dataset with specified number of aligned and conflicting samples per class.
    Dataset must have `align_indices` and `conflict_indices` attributes.
    """
    assert (rho is not None) ^ (align_count is not None and conflict_count is not None)

    # calculate the indices of aligned and conflicting samples per class
    align_indices = train_dataset.align_indices
    conflict_indices = train_dataset.conflict_indices
    n_align = [len(idxs) for idxs in align_indices.values()]
    n_conf = [len(idxs) for idxs in conflict_indices.values()]
    n_cl = [n_align[i] + n_conf[i] for i in range(2)]

    # calculate the number of aligned and conflicting samples to keep per class
    if rho is not None:  # Use the provided rho
        obj_n_align = [int(n_cl[i] * rho) for i in range(2)]
        obj_n_conf = [n_cl[i] - obj_n_align[i] for i in range(2)]
        actual_align_count = min(min(n_align), min(obj_n_align))
        actual_conflict_count = min(min(n_conf), min(obj_n_conf))
        actual_rho = actual_align_count / (actual_align_count + actual_conflict_count)
        if actual_rho > rho:  # too much aligned samples
            conflict_count = actual_conflict_count
            # r = a/(a+c) =>  a = rc/(1-r)
            align_count = int(conflict_count * rho / (1 - rho))
        elif actual_rho < rho:  # too much conflicting samples
            align_count = actual_align_count
            # r = a/(a+c)  => c = (1-r)a/r
            conflict_count = int(align_count * (1 - rho) / rho)
        else:  # just right
            align_count = actual_align_count
            conflict_count = actual_conflict_count
        effective_rho = align_count / (align_count + conflict_count)
        print(
            f"Using rho={rho} to set align_count={align_count} and conflict_count={conflict_count} (per class)"
        )
        print(f"Effective rho: {effective_rho}")
    else:  # Use the provided align_count and conflict_count
        if align_count > min(n_align):
            print(f"{align_count} exceeds available aligned samples.")
            align_count = min(n_align)
            print(f"Using {align_count} instead.")
        if conflict_count > min(n_conf):
            print(f"{conflict_count} exceeds available conflicting samples.")
            conflict_count = min(n_conf)
            print(f"Using {conflict_count} instead.")
        print(
            f"Using {align_count} aligned and {conflict_count} conflicting samples (per class)."
        )
        print(f"Effective rho: {align_count / (align_count + conflict_count)}")

    # Select balanced samples for each class
    selected_indices = []
    for class_label in range(2):
        # Get align samplesed
        selected_align = torch.randperm(n_align[class_label])[:align_count]
        selected_indices.extend(
            [align_indices[class_label][idx] for idx in selected_align]
        )

        # Get conflicting samples
        selected_conflict = torch.randperm(n_conf[class_label])[:conflict_count]
        selected_indices.extend(
            [conflict_indices[class_label][i] for i in selected_conflict]
        )

    # Create the balanced subset using our custom BalancedSubset class
    balanced_dataset = torch.utils.data.Subset(train_dataset, selected_indices)

    return balanced_dataset
