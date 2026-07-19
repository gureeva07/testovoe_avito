def compute_ap_at_10(predicted: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0

    hits = 0
    score = 0.0

    for i, article_id in enumerate(predicted[:10], start=1):
        if article_id in relevant:
            hits += 1
            # precision@i — сколько правильных среди первых i штук
            score += hits / i

    # делю на количество правильных ответов (но не больше 10) чтобы нормировать в 0..1
    return score / min(len(relevant), 10)


def compute_map_at_10(calibration, search_fn) -> float:
    # считаю MAP@10 по всем запросам из calibration
    ap_scores = []

    for _, row in calibration.iterrows():
        answer_str = search_fn(row["query_text"])
        predicted = [int(x) for x in answer_str.split()]
        relevant = {int(x) for x in str(row["ground_truth"]).split()}
        ap = compute_ap_at_10(predicted, relevant)
        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores)
