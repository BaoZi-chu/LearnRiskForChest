import numpy as np


def forward_sim_representation(model, input):
    r"""Performs a forward pass through the model.

    Overrides :meth:`torch.nn.Module.forward`.

    Args:
        model: class:`~deepmatcher.core.MatchingModel`.
        input: (:class:`~deepmatcher.batch.MatchingBatch`): A batch of tuple pairs
            processed into tensors.
    """
    embeddings = {}
    for name in model.meta.all_text_fields:
        attr_input = getattr(input, name)
        embeddings[name] = model.embed[name](attr_input)

    attr_comparisons = []
    for name in model.meta.canonical_text_fields:
        left, right = model.meta.text_fields[name]
        left_summary, right_summary = model.attr_summarizers[name](embeddings[left],
                                                                   embeddings[right])

        # Remove metadata information at this point.
        left_summary, right_summary = left_summary.data, right_summary.data

        if model.attr_condensors:
            left_summary = model.attr_condensors[name](left_summary)
            right_summary = model.attr_condensors[name](right_summary)
        attr_comparisons.append(model.attr_comparators[name](left_summary,
                                                             right_summary))
    return attr_comparisons


def get_attr_sim_representation(model,
                                dataset,
                                train=False,
                                device=None,
                                batch_size=32,
                                sort_in_buckets=None
                                ):
    from deepmatcher.data import MatchingIterator
    sort_in_buckets = train
    run_iter = MatchingIterator(
        dataset,
        model.meta,
        train,
        batch_size=batch_size,
        device=device,
        sort_in_buckets=sort_in_buckets)

    data_ids = []
    results = None
    for batch_idx, batch in enumerate(run_iter):
        attr_comparisons = forward_sim_representation(model, batch)
        data_ids.extend(batch.id)
        data_attr_matrix = attr_comparisons[0].data.numpy()
        for i in range(1, len(attr_comparisons)):
            data_attr_matrix = np.concatenate((data_attr_matrix, attr_comparisons[i].data.numpy()),
                                              axis=1)
        if results is None:
            results = data_attr_matrix
        else:
            results = np.concatenate((results, data_attr_matrix), axis=0)
    data_ids = np.array(data_ids)
    return data_ids, results

