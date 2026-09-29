# Resultados iniciais por método e dataset

Fonte: `/home/gab04/.codex/attachments/425de9a8-5cf1-4130-a4fe-52d9bc5efa73/pasted-text.txt`

## Resumo por método

| Método | Datasets | K ok | K falhou | ARI HDBSCAN médio | HAI médio | Tempo médio (s) |
| --- | --- | --- | --- | --- | --- | --- |
| HDBSCAN | 41 | 2009 | 0 | 1.0000 | 1.0000 | 9.972 |
| Optimized HDBSCAN | 41 | 2009 | 0 | 1.0000 | 1.0000 | 11.137 |
| ScoreSG | 41 | 0 | 2009 | - | - | 3.365 |
| ScoreSG Random | 41 | 0 | 2009 | - | - | 0.000 |

## Tabela por método e dataset

| Dataset | N | Features | Método | K ok | K falhou | ARI HDBSCAN médio | ARI HDBSCAN mín. | HAI médio | HAI mín. | Tempo médio (s) | Tempo total (s) | Clusters médios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian-iid-n10000-d10-seed42 | 10000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.886 | 190.406 | 0.3265 |
| gaussian-iid-n10000-d10-seed42 | 10000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.907 | 142.458 | 0.3265 |
| gaussian-iid-n10000-d10-seed42 | 10000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 1.926 | 1.926 | - |
| gaussian-iid-n10000-d10-seed42 | 10000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n10000-d128-seed42 | 10000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.273 | 160.392 | 0.1837 |
| gaussian-iid-n10000-d128-seed42 | 10000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 22.693 | 1111.968 | 0.1837 |
| gaussian-iid-n10000-d128-seed42 | 10000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 2.794 | 2.794 | - |
| gaussian-iid-n10000-d128-seed42 | 10000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n10000-d2-seed42 | 10000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.060 | 149.960 | 87.4694 |
| gaussian-iid-n10000-d2-seed42 | 10000 | 2 | Optimized HDBSCAN | 49 | 0 | 0.9996 | 0.9834 | 1.0000 | 1.0000 | 0.939 | 46.020 | 87.3673 |
| gaussian-iid-n10000-d2-seed42 | 10000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 1.730 | 1.730 | - |
| gaussian-iid-n10000-d2-seed42 | 10000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n10000-d20-seed42 | 10000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.604 | 176.591 | 0.2245 |
| gaussian-iid-n10000-d20-seed42 | 10000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.921 | 192.115 | 0.2245 |
| gaussian-iid-n10000-d20-seed42 | 10000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 1.998 | 1.998 | - |
| gaussian-iid-n10000-d20-seed42 | 10000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n10000-d32-seed42 | 10000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.445 | 168.828 | 0.1633 |
| gaussian-iid-n10000-d32-seed42 | 10000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.966 | 292.333 | 0.1633 |
| gaussian-iid-n10000-d32-seed42 | 10000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 2.098 | 2.098 | - |
| gaussian-iid-n10000-d32-seed42 | 10000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n10000-d64-seed42 | 10000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.349 | 164.084 | 0.3061 |
| gaussian-iid-n10000-d64-seed42 | 10000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.356 | 507.420 | 0.3061 |
| gaussian-iid-n10000-d64-seed42 | 10000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 2.275 | 2.275 | - |
| gaussian-iid-n10000-d64-seed42 | 10000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d10-seed42 | 20000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 17.964 | 880.221 | 0.3469 |
| gaussian-iid-n20000-d10-seed42 | 20000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11.179 | 547.768 | 0.3469 |
| gaussian-iid-n20000-d10-seed42 | 20000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 4.234 | 4.234 | - |
| gaussian-iid-n20000-d10-seed42 | 20000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d128-seed42 | 20000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.384 | 753.805 | 0.1429 |
| gaussian-iid-n20000-d128-seed42 | 20000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 96.084 | 4708.125 | 0.1429 |
| gaussian-iid-n20000-d128-seed42 | 20000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 5.801 | 5.801 | - |
| gaussian-iid-n20000-d128-seed42 | 20000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d2-seed42 | 20000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.427 | 706.905 | 169.1633 |
| gaussian-iid-n20000-d2-seed42 | 20000 | 2 | Optimized HDBSCAN | 49 | 0 | 0.9997 | 0.9901 | 1.0000 | 1.0000 | 3.388 | 166.030 | 169.1020 |
| gaussian-iid-n20000-d2-seed42 | 20000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 3.507 | 3.507 | - |
| gaussian-iid-n20000-d2-seed42 | 20000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d20-seed42 | 20000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.923 | 829.240 | 0.2449 |
| gaussian-iid-n20000-d20-seed42 | 20000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.232 | 746.383 | 0.2449 |
| gaussian-iid-n20000-d20-seed42 | 20000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 4.326 | 4.326 | - |
| gaussian-iid-n20000-d20-seed42 | 20000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d32-seed42 | 20000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 16.222 | 794.855 | 0.2041 |
| gaussian-iid-n20000-d32-seed42 | 20000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 23.039 | 1128.924 | 0.2041 |
| gaussian-iid-n20000-d32-seed42 | 20000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 4.516 | 4.516 | - |
| gaussian-iid-n20000-d32-seed42 | 20000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n20000-d64-seed42 | 20000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 15.588 | 763.833 | 0.1837 |
| gaussian-iid-n20000-d64-seed42 | 20000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 42.157 | 2065.677 | 0.1837 |
| gaussian-iid-n20000-d64-seed42 | 20000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 4.928 | 4.928 | - |
| gaussian-iid-n20000-d64-seed42 | 20000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n30000-d10-seed42 | 30000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 40.251 | 1972.318 | 0.6122 |
| gaussian-iid-n30000-d10-seed42 | 30000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 25.099 | 1229.860 | 0.6122 |
| gaussian-iid-n30000-d10-seed42 | 30000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 6.645 | 6.645 | - |
| gaussian-iid-n30000-d10-seed42 | 30000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n30000-d2-seed42 | 30000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 33.084 | 1621.110 | 250.3673 |
| gaussian-iid-n30000-d2-seed42 | 30000 | 2 | Optimized HDBSCAN | 49 | 0 | 0.9996 | 0.9910 | 1.0000 | 1.0000 | 7.274 | 356.405 | 250.2857 |
| gaussian-iid-n30000-d2-seed42 | 30000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 5.397 | 5.397 | - |
| gaussian-iid-n30000-d2-seed42 | 30000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n30000-d20-seed42 | 30000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 38.405 | 1881.847 | 0.2449 |
| gaussian-iid-n30000-d20-seed42 | 30000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 34.401 | 1685.643 | 0.2449 |
| gaussian-iid-n30000-d20-seed42 | 30000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 7.032 | 7.032 | - |
| gaussian-iid-n30000-d20-seed42 | 30000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d10-seed42 | 5000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.872 | 42.750 | 0.2449 |
| gaussian-iid-n5000-d10-seed42 | 5000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.744 | 36.459 | 0.2449 |
| gaussian-iid-n5000-d10-seed42 | 5000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 0.938 | 0.938 | - |
| gaussian-iid-n5000-d10-seed42 | 5000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d128-seed42 | 5000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.732 | 35.858 | 0.1837 |
| gaussian-iid-n5000-d128-seed42 | 5000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.708 | 279.697 | 0.1837 |
| gaussian-iid-n5000-d128-seed42 | 5000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 1.249 | 1.249 | - |
| gaussian-iid-n5000-d128-seed42 | 5000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d2-seed42 | 5000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.713 | 34.937 | 33.8163 |
| gaussian-iid-n5000-d2-seed42 | 5000 | 2 | Optimized HDBSCAN | 49 | 0 | 0.9998 | 0.9924 | 1.0000 | 1.0000 | 0.275 | 13.470 | 33.8163 |
| gaussian-iid-n5000-d2-seed42 | 5000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 19.551 | 19.551 | - |
| gaussian-iid-n5000-d2-seed42 | 5000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d20-seed42 | 5000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.820 | 40.203 | 0.2449 |
| gaussian-iid-n5000-d20-seed42 | 5000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.002 | 49.078 | 0.2449 |
| gaussian-iid-n5000-d20-seed42 | 5000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 0.958 | 0.958 | - |
| gaussian-iid-n5000-d20-seed42 | 5000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d32-seed42 | 5000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.781 | 38.292 | 0.1633 |
| gaussian-iid-n5000-d32-seed42 | 5000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.516 | 74.294 | 0.1633 |
| gaussian-iid-n5000-d32-seed42 | 5000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 0.992 | 0.992 | - |
| gaussian-iid-n5000-d32-seed42 | 5000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-iid-n5000-d64-seed42 | 5000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.751 | 36.812 | 0.2857 |
| gaussian-iid-n5000-d64-seed42 | 5000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.718 | 133.193 | 0.2857 |
| gaussian-iid-n5000-d64-seed42 | 5000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 1.076 | 1.076 | - |
| gaussian-iid-n5000-d64-seed42 | 5000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d10-seed42 | 10000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.079 | 150.854 | 6.0000 |
| gaussian-separated-n10000-d10-seed42 | 10000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.269 | 62.197 | 6.0000 |
| gaussian-separated-n10000-d10-seed42 | 10000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 1.765 | 1.765 | - |
| gaussian-separated-n10000-d10-seed42 | 10000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d128-seed42 | 10000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.031 | 148.532 | 6.0000 |
| gaussian-separated-n10000-d128-seed42 | 10000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.968 | 537.433 | 6.0000 |
| gaussian-separated-n10000-d128-seed42 | 10000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 2.404 | 2.404 | - |
| gaussian-separated-n10000-d128-seed42 | 10000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d2-seed42 | 10000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.999 | 146.948 | 49.5306 |
| gaussian-separated-n10000-d2-seed42 | 10000 | 2 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.858 | 42.020 | 49.5306 |
| gaussian-separated-n10000-d2-seed42 | 10000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 1.725 | 1.725 | - |
| gaussian-separated-n10000-d2-seed42 | 10000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d20-seed42 | 10000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.033 | 148.609 | 6.0000 |
| gaussian-separated-n10000-d20-seed42 | 10000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.811 | 88.739 | 6.0000 |
| gaussian-separated-n10000-d20-seed42 | 10000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 1.859 | 1.859 | - |
| gaussian-separated-n10000-d20-seed42 | 10000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d32-seed42 | 10000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.019 | 147.916 | 6.0000 |
| gaussian-separated-n10000-d32-seed42 | 10000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.765 | 135.504 | 6.0000 |
| gaussian-separated-n10000-d32-seed42 | 10000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 1.920 | 1.920 | - |
| gaussian-separated-n10000-d32-seed42 | 10000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n10000-d64-seed42 | 10000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.010 | 147.492 | 6.0000 |
| gaussian-separated-n10000-d64-seed42 | 10000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5.036 | 246.740 | 6.0000 |
| gaussian-separated-n10000-d64-seed42 | 10000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 2.071 | 2.071 | - |
| gaussian-separated-n10000-d64-seed42 | 10000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d10-seed42 | 20000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.853 | 727.811 | 6.0000 |
| gaussian-separated-n20000-d10-seed42 | 20000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.922 | 241.180 | 6.0000 |
| gaussian-separated-n20000-d10-seed42 | 20000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 3.919 | 3.919 | - |
| gaussian-separated-n20000-d10-seed42 | 20000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d128-seed42 | 20000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.507 | 710.834 | 6.0000 |
| gaussian-separated-n20000-d128-seed42 | 20000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 47.438 | 2324.452 | 6.0000 |
| gaussian-separated-n20000-d128-seed42 | 20000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 5.003 | 5.003 | - |
| gaussian-separated-n20000-d128-seed42 | 20000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d2-seed42 | 20000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.210 | 696.285 | 110.1020 |
| gaussian-separated-n20000-d2-seed42 | 20000 | 2 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.160 | 154.853 | 110.1020 |
| gaussian-separated-n20000-d2-seed42 | 20000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 3.495 | 3.495 | - |
| gaussian-separated-n20000-d2-seed42 | 20000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d20-seed42 | 20000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.671 | 718.894 | 6.0000 |
| gaussian-separated-n20000-d20-seed42 | 20000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 6.956 | 340.837 | 6.0000 |
| gaussian-separated-n20000-d20-seed42 | 20000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 3.872 | 3.872 | - |
| gaussian-separated-n20000-d20-seed42 | 20000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d32-seed42 | 20000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.457 | 708.396 | 6.0000 |
| gaussian-separated-n20000-d32-seed42 | 20000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.883 | 533.246 | 6.0000 |
| gaussian-separated-n20000-d32-seed42 | 20000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 4.214 | 4.214 | - |
| gaussian-separated-n20000-d32-seed42 | 20000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n20000-d64-seed42 | 20000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 14.404 | 705.797 | 6.0000 |
| gaussian-separated-n20000-d64-seed42 | 20000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 20.044 | 982.155 | 6.0000 |
| gaussian-separated-n20000-d64-seed42 | 20000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 4.420 | 4.420 | - |
| gaussian-separated-n20000-d64-seed42 | 20000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n30000-d10-seed42 | 30000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 33.493 | 1641.155 | 6.0000 |
| gaussian-separated-n30000-d10-seed42 | 30000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10.947 | 536.409 | 6.0000 |
| gaussian-separated-n30000-d10-seed42 | 30000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 6.065 | 6.065 | - |
| gaussian-separated-n30000-d10-seed42 | 30000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n30000-d2-seed42 | 30000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 32.527 | 1593.819 | 167.5918 |
| gaussian-separated-n30000-d2-seed42 | 30000 | 2 | Optimized HDBSCAN | 49 | 0 | 0.9998 | 0.9881 | 1.0000 | 1.0000 | 6.896 | 337.921 | 167.5306 |
| gaussian-separated-n30000-d2-seed42 | 30000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 5.524 | 5.524 | - |
| gaussian-separated-n30000-d2-seed42 | 30000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d10-seed42 | 5000 | 10 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.678 | 33.238 | 6.0000 |
| gaussian-separated-n5000-d10-seed42 | 5000 | 10 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.348 | 17.050 | 6.0000 |
| gaussian-separated-n5000-d10-seed42 | 5000 | 10 | ScoreSG | 0 | 49 | - | - | - | - | 0.867 | 0.867 | - |
| gaussian-separated-n5000-d10-seed42 | 5000 | 10 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d128-seed42 | 5000 | 128 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.671 | 32.885 | 6.0000 |
| gaussian-separated-n5000-d128-seed42 | 5000 | 128 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.888 | 141.519 | 6.0000 |
| gaussian-separated-n5000-d128-seed42 | 5000 | 128 | ScoreSG | 0 | 49 | - | - | - | - | 1.182 | 1.182 | - |
| gaussian-separated-n5000-d128-seed42 | 5000 | 128 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d2-seed42 | 5000 | 2 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.669 | 32.778 | 22.6939 |
| gaussian-separated-n5000-d2-seed42 | 5000 | 2 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.245 | 11.981 | 22.6939 |
| gaussian-separated-n5000-d2-seed42 | 5000 | 2 | ScoreSG | 0 | 49 | - | - | - | - | 0.855 | 0.855 | - |
| gaussian-separated-n5000-d2-seed42 | 5000 | 2 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d20-seed42 | 5000 | 20 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.671 | 32.871 | 6.0000 |
| gaussian-separated-n5000-d20-seed42 | 5000 | 20 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.481 | 23.558 | 6.0000 |
| gaussian-separated-n5000-d20-seed42 | 5000 | 20 | ScoreSG | 0 | 49 | - | - | - | - | 0.902 | 0.902 | - |
| gaussian-separated-n5000-d20-seed42 | 5000 | 20 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d32-seed42 | 5000 | 32 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.664 | 32.523 | 6.0000 |
| gaussian-separated-n5000-d32-seed42 | 5000 | 32 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.738 | 36.174 | 6.0000 |
| gaussian-separated-n5000-d32-seed42 | 5000 | 32 | ScoreSG | 0 | 49 | - | - | - | - | 0.913 | 0.913 | - |
| gaussian-separated-n5000-d32-seed42 | 5000 | 32 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
| gaussian-separated-n5000-d64-seed42 | 5000 | 64 | HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.668 | 32.716 | 6.0000 |
| gaussian-separated-n5000-d64-seed42 | 5000 | 64 | Optimized HDBSCAN | 49 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.366 | 66.940 | 6.0000 |
| gaussian-separated-n5000-d64-seed42 | 5000 | 64 | ScoreSG | 0 | 49 | - | - | - | - | 1.007 | 1.007 | - |
| gaussian-separated-n5000-d64-seed42 | 5000 | 64 | ScoreSG Random | 0 | 49 | - | - | - | - | 0.000 | 0.000 | - |
