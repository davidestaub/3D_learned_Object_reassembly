from scipy.special import factorial


def get_viewer_data(fragments=None, keypoints=None):
    data = {}
    if fragments:
        data["fragments"] = fragments

    if keypoints:
        data["keypoints"] = keypoints
    return data


def nchoosek(n, k) -> int:
    return factorial(n) / (factorial(n - k) * factorial(k))
