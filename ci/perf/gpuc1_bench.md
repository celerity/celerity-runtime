# Benchmark Results

| Metadata |                      |
| :------- | :------------------- |
| Created  | 2022-03-29T13:58:21Z |


| Test case                                                                                                                                         | Benchmark name                   |           Mean |       Std dev |
| :------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------- | -------------: | ------------: |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | creating nodes                   |           3.39 |          0.11 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | creating and adding dependencies |          24.74 |          0.66 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | adding and removing dependencies |          17.02 |          1.03 |
| benchmark intrusive graph dependency handling with N nodes - 1                                                                                    | checking for dependencies        |           2.10 |          0.03 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | creating nodes                   |          40.40 |          0.45 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | creating and adding dependencies |         280.19 |          8.26 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | adding and removing dependencies |         240.21 |          4.94 |
| benchmark intrusive graph dependency handling with N nodes - 10                                                                                   | checking for dependencies        |          45.12 |          0.85 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | creating nodes                   |         386.25 |         16.67 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | creating and adding dependencies |       4'654.40 |         17.62 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | adding and removing dependencies |       4'744.76 |         15.56 |
| benchmark intrusive graph dependency handling with N nodes - 100                                                                                  | checking for dependencies        |       2'003.76 |         48.21 |
| generating large task graphs                                                                                                                      | soup topology                    |  10'714'650.81 |    675'713.74 |
| generating large task graphs                                                                                                                      | chain topology                   |      70'865.68 |      1'440.36 |
| generating large task graphs                                                                                                                      | expanding tree topology          |     116'375.89 |      1'130.40 |
| generating large task graphs                                                                                                                      | contracting tree topology        |     162'857.78 |     12'385.56 |
| generating large task graphs                                                                                                                      | wave_sim topology                |     651'681.74 |     40'443.94 |
| generating large task graphs                                                                                                                      | jacobi topology                  |     212'690.90 |     16'683.59 |
| generating large command graphs for N nodes - 1                                                                                                   | soup topology                    |  17'469'026.62 |    944'443.84 |
| generating large command graphs for N nodes - 1                                                                                                   | chain topology                   |     279'363.33 |      2'300.96 |
| generating large command graphs for N nodes - 1                                                                                                   | expanding tree topology          |     379'904.05 |     17'332.99 |
| generating large command graphs for N nodes - 1                                                                                                   | contracting tree topology        |     482'917.06 |      4'068.51 |
| generating large command graphs for N nodes - 1                                                                                                   | wave_sim topology                |   2'224'281.25 |      6'339.61 |
| generating large command graphs for N nodes - 1                                                                                                   | jacobi topology                  |     814'997.50 |      3'844.40 |
| generating large command graphs for N nodes - 4                                                                                                   | soup topology                    |  42'768'934.41 |  1'133'510.78 |
| generating large command graphs for N nodes - 4                                                                                                   | chain topology                   |   3'268'356.09 |    117'234.13 |
| generating large command graphs for N nodes - 4                                                                                                   | expanding tree topology          |   6'166'049.95 |    343'557.44 |
| generating large command graphs for N nodes - 4                                                                                                   | contracting tree topology        |   3'661'062.41 |    145'209.92 |
| generating large command graphs for N nodes - 4                                                                                                   | wave_sim topology                |  14'922'201.50 |    568'393.60 |
| generating large command graphs for N nodes - 4                                                                                                   | jacobi topology                  |   5'049'773.16 |    269'112.22 |
| generating large command graphs for N nodes - 16                                                                                                  | soup topology                    | 146'092'649.41 |  2'243'339.63 |
| generating large command graphs for N nodes - 16                                                                                                  | chain topology                   | 383'212'138.32 | 10'716'940.63 |
| generating large command graphs for N nodes - 16                                                                                                  | expanding tree topology          | 398'452'871.51 |  8'096'621.98 |
| generating large command graphs for N nodes - 16                                                                                                  | contracting tree topology        | 125'153'408.68 |  2'392'983.59 |
| generating large command graphs for N nodes - 16                                                                                                  | wave_sim topology                | 127'183'654.92 |  2'262'326.46 |
| generating large command graphs for N nodes - 16                                                                                                  | jacobi topology                  | 117'483'912.89 |  3'590'756.69 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | soup topology                    |  18'476'474.89 |    787'846.36 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | chain topology                   |     276'750.00 |      2'087.09 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | expanding tree topology          |     392'546.64 |      3'250.50 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | contracting tree topology        |     489'482.07 |      3'747.41 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | wave_sim topology                |   2'136'943.57 |    109'940.33 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: single-threaded immediate graph generation                   | jacobi topology                  |     824'098.32 |      4'431.16 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | soup topology                    |  36'930'442.41 |  2'860'011.53 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | chain topology                   |     812'496.03 |    193'119.67 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | expanding tree topology          |     885'353.46 |     77'674.17 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | contracting tree topology        |   1'422'068.06 |    275'497.34 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | wave_sim topology                |   6'444'928.86 |  1'221'588.14 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > immediate submission to a scheduler thread                              | jacobi topology                  |   2'041'584.46 |    321'299.39 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | soup topology                    |  20'959'286.12 |    523'863.55 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | chain topology                   |     584'500.38 |     19'650.52 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | expanding tree topology          |     703'885.65 |      3'626.47 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | contracting tree topology        |     804'693.94 |      5'824.86 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | wave_sim topology                |   4'283'792.75 |     87'487.77 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > reference: throttled single-threaded graph generation at 10 us per task | jacobi topology                  |   1'338'703.22 |      9'416.29 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | soup topology                    |  31'987'835.49 |  7'656'126.77 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | chain topology                   |   1'386'185.74 |    205'961.74 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | expanding tree topology          |   1'536'617.78 |    311'032.75 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | contracting tree topology        |   1'571'582.53 |    297'591.65 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | wave_sim topology                |  10'108'475.96 |  1'777'042.37 |
| building command graphs in a dedicated scheduler thread for N nodes - 1 > throttled submission to a scheduler thread at 10 us per task            | jacobi topology                  |   2'163'554.30 |    761'256.55 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | soup topology                    |  42'280'877.58 |  1'806'547.23 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | chain topology                   |   3'330'472.14 |    151'589.26 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | expanding tree topology          |   6'590'028.70 |    101'953.81 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | contracting tree topology        |   3'777'119.73 |      7'693.48 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | wave_sim topology                |  15'474'509.88 |    383'219.42 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: single-threaded immediate graph generation                   | jacobi topology                  |   5'201'217.64 |    247'854.97 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | soup topology                    |  72'927'489.66 |  4'029'350.97 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | chain topology                   |   4'668'623.13 |    895'811.79 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | expanding tree topology          |   7'760'416.98 |  1'054'051.39 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | contracting tree topology        |   4'777'635.68 |  1'009'106.57 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | wave_sim topology                |  25'646'671.39 |  4'196'259.80 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > immediate submission to a scheduler thread                              | jacobi topology                  |   9'779'632.80 |  1'365'076.60 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | soup topology                    |  44'745'879.66 |  1'866'700.99 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | chain topology                   |   3'572'231.56 |    261'092.75 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | expanding tree topology          |   6'668'761.62 |    399'940.27 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | contracting tree topology        |   3'760'732.90 |    199'535.14 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | wave_sim topology                |  16'725'383.05 |    570'891.44 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > reference: throttled single-threaded graph generation at 10 us per task | jacobi topology                  |   5'500'763.84 |    280'110.75 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | soup topology                    |  58'796'896.47 | 18'899'214.28 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | chain topology                   |   6'220'595.79 |  1'482'269.45 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | expanding tree topology          |  11'363'326.38 |  1'330'639.95 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | contracting tree topology        |   3'720'665.49 |    146'938.57 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | wave_sim topology                |  29'573'964.49 |  5'178'408.35 |
| building command graphs in a dedicated scheduler thread for N nodes - 4 > throttled submission to a scheduler thread at 10 us per task            | jacobi topology                  |  10'017'942.86 |  1'925'964.37 |

All numbers are in nanoseconds.
