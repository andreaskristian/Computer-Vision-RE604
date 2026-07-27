"""
cer_utils.py
------------
Character Error Rate (CER) calculation.

CER = (S + D + I) / N
  S = jumlah karakter substitusi (salah)
  D = jumlah karakter yang dihapus (ada di ground truth, hilang di prediksi)
  I = jumlah karakter yang disisipkan (ada di prediksi, tidak ada di ground truth)
  N = jumlah karakter pada ground truth

Dihitung menggunakan Levenshtein edit-distance (dynamic programming),
lalu di-backtrace untuk memisahkan jumlah S, D, dan I secara terpisah
(bukan hanya total edit distance).
"""


def compute_cer(ground_truth: str, prediction: str):
    """
    Menghitung CER antara ground_truth dan prediction.

    Returns:
        cer (float), S (int), D (int), I (int)
    """
    gt = ground_truth or ""
    pred = prediction or ""

    n = len(gt)
    m = len(pred)

    # DP table utk edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    # Backtrace untuk memisahkan S, D, I
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt[i - 1] == pred[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1
            j -= 1
        else:
            # fallback safety (should not happen)
            break

    N = len(gt) if len(gt) > 0 else 1  # hindari pembagian dengan 0
    cer = (S + D + I) / N
    return cer, S, D, I


if __name__ == "__main__":
    # Quick self-test
    tests = [
        ("BP1234CD", "BP1234CD", 0.0),   # sempurna
        ("BP1234CD", "BP1234C", None),   # 1 deletion
        ("BP1234CD", "BP1234CDX", None), # 1 insertion
        ("BP1234CD", "BP1284CD", None),  # 1 substitution
        ("", "ABC", None),               # ground truth kosong
    ]
    for gt, pred, expected in tests:
        cer, S, D, I = compute_cer(gt, pred)
        print(f"GT='{gt}' PRED='{pred}' -> CER={cer:.4f} (S={S}, D={D}, I={I})")
