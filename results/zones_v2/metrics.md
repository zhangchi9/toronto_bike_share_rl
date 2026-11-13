# Zone schemes compared

Mainland stations only. Road min = mean drive time between two stations of the same zone. Profile corr = mean correlation of two same-zone stations' hourly net-flow-per-dock profiles. Variance explained = share of station profile variance captured by zone means. Isolated = none of the 4 nearest stations by road is in the same zone.

| scheme | zones | docks min / median / max | docks CV | stations min-max | road min within zone | profile corr within zone | profile variance explained | isolated stations |
|---|---|---|---|---|---|---|---|---|
| kmeans (rounds 1-4) | 20 | 164 / 728 / 3,229 | 0.77 | 12-135 | 4.1 | 0.09 | 11% | 7 |
| flow w=0.3 (30% road, 70% demand) | 21 | 909 / 1,015 / 1,148 | 0.09 | 32-73 | 4.6 | 0.26 | 41% | 17 |
| flow w=0.5 (50% road, 50% demand) | 21 | 907 / 1,023 / 1,148 | 0.09 | 32-78 | 4.3 | 0.24 | 39% | 13 |
| flow w=0.7 (70% road, 30% demand) | 21 | 915 / 1,029 / 1,151 | 0.08 | 40-62 | 4.0 | 0.23 | 37% | 3 |
