### SFT Data Distribution

| Family                 | Unique QA      | QA share         | Turns           | Turn share |
| ---------------------- | -------------- | ---------------- | --------------- | ---------- |
| cgbench                | 2930           | **35.88%** | 10643           | 33.15%     |
| longvideobench-imitate | 2304           | **28.22%** | 10183           | 31.72%     |
| llava-academic         | 1387           | **16.99%** | 5410            | 16.85%     |
| llava-youtube          | 1298           | **15.90%** | 5106            | 15.91%     |
| videoholmes-train      | 246            | **3.01%**  | 759             | 2.36%      |
| **Total**        | **8165** | 100%             | **32101** | 100%       |

## GRPO Selection

Filtering rule: run pass@8 (T=0.7) on `train.parquet`, keep samples with pass ∈ [1, 6].

| Family                 | Original        | Selected       | Selected share   | Retention rate |
| ---------------------- | --------------- | -------------- | ---------------- | -------------- |
| cgbench                | 3593            | 2241           | **48.06%** | 62.37%         |
| llava-youtube          | 3063            | 1079           | **23.14%** | 35.23%         |
| llava-academic         | 2476            | 793            | **17.01%** | 32.03%         |
| videoholmes-train      | 489             | 314            | **6.73%**  | 64.21%         |
| longvideobench-imitate | 400             | 236            | **5.06%**  | 59.00%         |
| **Total**        | **10021** | **4663** | 100%             | 46.53%         |
