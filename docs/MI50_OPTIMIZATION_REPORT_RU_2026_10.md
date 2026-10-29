# MI50: журнал оптимизации DeepSeek V4.1 Flash, 27–29 октября 2026

## Что получено и чего этот отчёт не доказывает

Рабочий кандидат объединяет YOCO, распределение trunk-слоёв 10/30 и сужающийся хвост (cone). Код комбинации: `8a955294ec7a61752e6f6b5e189dd3203b93c89f`, база — опубликованный YOCO15/25 `c4a85a825854912ebbc1a5b18c5f7d1eb012ae6f`. В одном целевом запуске 20 160 токенов с пустым prefix cache prefill занял **58,7005 с**, TTFT — **58,7894 с**. Полный HTTP-запрос с коротким ответом — 59,0621 с. Это измерение комбинации, а не сумма процентов отдельных оптимизаций.

Численная эквивалентность исходной версии НЕ доказана. Изменение числа строк и резидентности экспертов меняет существующий light/tiled dispatch и FP-траекторию. Пользователь после объяснения этого ограничения отдельно поручил собрать и оставить комбинацию работающей. Не менялись веса, количество выбираемых экспертов, target sampling, precision-флаги или политика KV. Однако это не означает побитовую идентичность: существующий backend сам использует разные представления активаций при разных размерах групп.

Публикуются код, узкие тесты и этот документ. Сырые запросы, logits, снимки памяти, профили пользователя, ключи и большие журналы остаются только на стенде. Пути evidence ниже — ссылки на локальные свидетельства, а не вложения репозитория. Другие GPU и CUDA/SYCL/Metal не проверялись. Это аппаратный эксперимент в отдельной feature-ветке, не запрос слияния в upstream.

## Стенд и неизменяемые условия

Проверено на Linux, двух AMD MI50 gfx906: **32 752 MiB VRAM на каждой** (34 342 961 152 байт). CPU — AMD EPYC 7K62, 48 физических ядер / 96 потоков, один socket и один NUMA node, CPUs 0–95. MemTotal — 527 996 548 kB, около 503,536 GiB доступной памяти при номинальных 512 GB.

Пак: DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF-iq2xs, каталог `maya-ds4-nvme-mtp`. Из его `ds4.txt`, а не из названия модели:

| Параметр | Значение |
| --- | ---: |
| Trunk / MTP слои | 40 / 3 |
| Hidden / mHC multiplicity | 5120 / 4 |
| Attention heads × head dimension / rope | 64 × 512 / 64 |
| q_lora / o_lora / o_groups | 1280 / 1024 / 8 |
| Expert FF / experts / picks | 2304 / 384 / 6 |
| Local window | 128 |
| Index heads × dim / top-k | 32 × 128 / 512 |
| Vocabulary | 129280 |
| KV sources | 2, 8, 14, 20 |
| Index owners | 2, 8, 14, 20, 24, 28, 32, 36 |
| Engram | слои 1, 14 |
| DSpark block / Markov rank | 5 / 256 |
| MTP experts / picks | 128 / 3 |
| DSpark taps | входы слоёв 37, 38, 39 |

Runtime context limit — 262144, pack limit — 1048576. Основной запрос имеет 20160 input IDs, capacity chunk 8192, фактические три равных блока по 6720, attention slab 256, indexer slab 32. SHA256 input IDs: `4fa8cea4b539f1268c5b9836888b45c0618381ec087bf36fff1a8b31dc3efa33`.

Слова «не добавлять квантование KV» не означают, что исходный KV был FP16. В этом pack/runtime CKV уже использует native E2M1 codes + E4M3 scales: для dim512 — 288 B/row. Index cache использует E2M1 + UE8M0 scales с выравниванием 16. Window cache упакован в FP8, рабочие строки — BF16. Эти форматы не изменены. Код: `src/dsv41/ds4_kernels.cuh`, `kv_row_bytes`, и создание CKV в `ds4_model.cu`.

Существующий step3 HIPK/fdot2 сохранён. Он включает FP16 округления scale/product; его нельзя называть побитовым эквивалентом scalar F32. Эта кампания не вводит дополнительного приближённого режима, но и не отменяет ранее выбранную арифметику.

## Как измеряли

Сравнивались prompt compute, TTFT и wall отдельно. Для prefill: одинаковые input IDs, cache_n=0, фиксированный исходный expert profile, неизменные HIPK, DSpark, sampler, context и chunk/slabs. Sampling целевого запроса: temperature0, top_p0.95, top_k20, seed42727. Сам по себе seed не устраняет все различия backend/streaming.

Для финальной комбинации временный main сохранял полный вектор logits после prefill и 11 векторов при восстановлении последнего токена с заменой и 160-token teacher-forced продолжении. Это диагностический main, не рабочая служба. Он и чистый main связаны с одним и тем же `model.o`. Время prefill не включает последующее диагностическое продолжение. Простой технический gate не заменяет оценку качества на корпусе.

Старый DS4_PROF суммировал интервалы по устройствам. Поэтому 151,4 с профиля нельзя делить на ~104 с wall и выдавать доли critical path. Кроме того, marker q_b+rope захватывал выходные проекции предыдущего slab. Для выбора баланса использовались GPU completion events и зависимости, а не CPU enqueue timestamps. Часы разных GPU напрямую не вычитались. ROCProfiler в прежней попытке дал “Context ID not found”; его результаты не использовались.

Формула throughput gain: `t_control / t_candidate - 1`. Сокращение времени — другая величина. Цели +30/+50% относительно 107,6017 с соответствуют 82,770538/71,734467 с. Ни отношение числа строк, ни сумма GPU peak не являются прогнозом ускорения.

## Хронология prefill: сопоставимые серии

| Серия | Контроль, с | Кандидат, с | Статус |
| --- | --- | --- | --- |
| Layer-local partial F16 cache, split21/19 | 107,6017 | 105,8145 | одна пара, +1,689%; выключен |
| YOCO15/25, оба варианта split15/25 | 114,5601 / 113,8569 | 74,4561 | два контроля, один кандидат |
| Cone на split15/25 | 76,5264 / 74,8917 | 70,2796 | один кандидат, численные отличия |
| Rectangular YOCO10/30 | 76,5171 / 76,3641 (15/25) | 63,0994 | один кандидат, +21,14% к среднему контролю |
| Финальный YOCO10/30 + cone | предыдущие 76,5171 / 76,3641 | 58,7005 | один новый запуск; контроль не повторялся одновременно |

Финальная комбинация на 30,22% быстрее по throughput среднего прежнего 15/25-контроля (76,4406 с) и на 7,49% быстрее прежнего rectangular10/30 (63,0994 с). Это сравнение с сохранённой серией при совпавших input IDs и исходном profile, **не новый чередующийся A/B**. От исторического 107,6017 с получается +83,31%, но там другое распределение и другая серия. Эти сравнения нельзя смешивать с доказательством повторяемости.

Ранние заявленные пользователем 147→193 tok/s относятся к другой серии step3; они не заменяют таблицу прямых измерений. В отдельном сохранённом прогоне 27 октября измерено 109,004 с на 20160. Короткие 8-token decode показатели из prefill-серий не являются steady decode benchmark.

## YOCO15/25: что было изменено

Реализация `badbd3c4504abe3a17eabbae6b249ab87fc7cb87`, публикация/инструкции `c4a85a825854912ebbc1a5b18c5f7d1eb012ae6f`.

Вместо запрета YOCO при двух устройствах добавлен guard конкретной топологии. GPU0 выполняет trunk0–14 и MTP40–42, GPU1 — trunk15–39 и head. CKV14 передаётся через границу; source20 локален GPU1. Два handoff slots имеют обязательные consume/ack. Нельзя просто пропустить поздний блок и не освободить slot: producer тогда может зависнуть.

Ранние слои и source20 history считаются для всех токенов. На поздних слоях сохраняется dependency tail2668. Для 20160 skip_prefix17492. L20 не удаляется целиком: его KV/index source-часть обязательна. YOCO не освобождает dense weights и capacity8192 workspace.

Передача свежего CKV14 для 20160: 10080 compressed rows ×288 B = 2 903 040 B. Дополнительная capacity CKV14 на принимающей GPU — 36,000488 MiB, с известными rings/tables —36,461914 MiB без dense/expert изменений.

В серии 15/25 полные финальные logits отличались: control/control max_abs1,98449, RMS0,455488; control/YOCO max_abs1,68606, RMS0,396283. Позже выяснилось, что ранние replay-контроли не восстанавливали весь adaptive profile/stream state. Поэтому эти числа НЕ доказывают неизбежную недетерминированность и не устанавливают качество.

## Cone: зависимости и ограничения

Файлы: `src/dsv41/ds4_yoco_pipeline.hpp` (`yoco_row_begin`), `ds4_model.cu` (`Run::r0`, `fl_rows`, `forward_chunks`, `run_dev_chunks`, suffix views в layer pass). Первичная реализация `35c5b6e44ec7dfb231d11df2d640ee388a312cb3`, milestone `6b60509b71fa1dcad42061f64a7ae0d6738c48c2`.

Пусть E — конец prompt, H_l=(h,pre). Для l=20…39:
`C_l = 2668 - 127*(l-20)`, `S_l = E-C_l`.
При правильных входах [S_l,E) межстрочная зависимость позднего слоя ограничена [t−127,t]. Первые 127 выходных строк могут зависеть от отброшенного halo; выходы [S_l+127,E) корректны по графу. Следующий слой начинается ровно с S_l+127. Последний слой получает 255 корректных входов и оставляет 128 корректных выходов.

Это доказательство графа, а не FP-identical execution. Полная CKV/IK history source20 обязательна. Query-row pointers topk/cand сдвигаются вместе с h/pre, но содержащиеся в них абсолютные compressed-key IDs не меняются. DSpark taps снимаются ДО слоя. Текущему кольцу нужны 128 входных строк, snapshot предыдущей позиции — объединение 129 строк; последние 255 входов L39 покрывают оба требования.

Cone opt-in ограничен поддержанным tail в последнем chunk. Для 20160 это chunk2, r0=4052. Вне условий — rectangular fallback. Один helper диапазонов обслуживает и реальные вычисления, и planner потоковой загрузки экспертов; иначе planner может загрузить неверный набор.

Точные work counts для поздних слоёв: row×layers53360→29230; attention slabs220→120 приA256; indexer rows13340→8260, slabs420→260 приAI32; token-expert assignments320160→175380. Уникальные загрузки экспертов не обязаны уменьшиться в той же пропорции.

### Почему изменились числа

На одном сохранённом post-L20 input у L21 q_b, RoPE, sparse attention и inverse RoPE совпали побитово. Первое отличие — wo_a: max_abs5,126e−6, RMS1,958e−7; после wo_b max_abs1,0401e−4, RMS1,1502e−5. Rectangular использовал slabs19796/256 и 20052/108, cone19923/237. Одинаковые rocBLAS GEMM формы восстановили bit equality L21 attention. Это подтверждает эффект формы в данном подядре, а не ошибку координат. Конкретный rocBLAS solution не записан. Сохранение исходных форм потребовало бы 139slabs/32624row-slots вместо 120/29230; цена wall не измерялась.

Replay первоначально восстанавливал численные буферы, но не весь expert profile, streaming origin и speculative counters. После исправления rectangular/rectangular стал bit-exact: 0 из 288 записей отличались. Старый разброс нельзя использовать как оправдание нового.

Далее L21 shared expert, routes, weights и порядок contributions совпали. Первое отличие routed GU связано с порогом 16 строк:
- light использует Q8 активации;
- tiled gfx906 HIPK использует F16/fdot2.
Реальные пары: token20106/expert227 rect16→cone17, GUmax0,0078125; token20135/expert30 rect18→cone16, GUmax0,01171875. Размеры групп не обязаны монотонно уменьшаться после изменения предшествующих вычислений.

Принудительное сохранение reference backend восстановило измеренные L21 значения, но позже возникло отличие L22 attention; final logits max0,629192/RMS0,119625. Для полного reference dispatch нужны маршруты отброшенных строк, а их вычисление не бесплатно. Общий дешёвый способ сохранить прежнюю траекторию не найден. Поэтому cone первоначально не внедрялся; позже пользователь отдельно разрешил комбинацию с раскрытым ограничением.

## Баланс 10/30 и VRAM

Диагностический trace рабочего YOCO15/25 дал 76,4464 с. Frozen-service DAG оценивал 10/30 в 63,4717 с, но предполагал неизменную стоимость services и не был benchmark. При переносе слоёв меняются residency и streamed bytes; расчёт проверен последующим реальным 63,0994 с.

Реализация 10/30: `dfc29cfc6e3a79291bf6a7d35dd8e457df80ba00`, milestone `ecc7c0523ff446f517c1a93d3473babd21b39ccb`.
GPU0: trunk0–9 + MTP40–42. GPU1: trunk10–39 + head.
CKV8 пересекает границу, IK8 GPU1 не нужен. Source14/20 и Engram14 находятся GPU1. L10–19 исполняются для каждого chunk, поэтому consume/ack сохраняется до позднего skip.

В комбинации `yoco_layer()` проверяет полную topology/physical allocation, затем `yoco_cone_pipeline_admitted()` допускает ровно один из проверенных pipeline10/pipeline15. Это не глобальное ослабление guard.

Startup ledger, байты (категории логов не следует считать полной разметкой каждого runtime allocation):

| Категория | 15/25 GPU0 | 15/25 GPU1 | 10/30 GPU0 | 10/30 GPU1 |
| --- | ---: | ---: | ---: | ---: |
| Weights | 3330088960 | 3782666496 | 2567010048 | 4545745408 |
| Experts | 26301542400 | 25905346560 | 27202775040 | 24916285440 |
| Caches | 145922304 | 135908608 | 97349120 | 184481792 |
| Scratch | 3360885248 | 3202521856 | 3360753408 | 3290209792 |
| Free | 612368384 | 597688320 | 610271232 | 597688320 |
| Physical slots | 2410 | 2625 | 2531 | 2533 |
| Resident experts | 2395 | 2600 | 2521 | 2503 |

Финальная комбинация подтвердила startup ledger10/30. Rings неизменны: GPU0 120×10229760B, GPU1 108×11335680B; C8192/A256/AI32. Startup free не является гарантией свободной памяти после late allocations. В предыдущей серии 10/30 free позднее был 358/344MiB против startup582/570MiB. В комбинации OOM не было.

На 10/30 GPU0 L0 имеет 221slots/220resident, L1–9 по 214/213; MTP128/128. GPU1 L10 имеет 97/96, L11–39 по 84/83. Нельзя умножать число slots на один общий размер: типы/размеры expert blobs различаются; ring stride — максимум, не размер всех resident blobs.

Перенос L10–14: dense payload763063552B; дополнительные компоненты включают CKV capacity37749248B, IK14 10486016B и Engram workspace87556096B GPU1. Наблюдённая разница nonexpert около−774,177/+857,677MiB; часть remainder runtime/allocator не разложена по тензорам. Это честнее, чем использовать свободную память из другого короткого запуска.

В прежней 10/30 серии logged streamed RAM: GPU0/GPU1 27,5/110,6GB против 71,2/60,4GB у 15/25; readwait650/2957ms против 1872/2151ms. Это агрегаты/proxy blob bytes, а не измеренный serial critical stall. Ordered expert-access trace для точного cache miss моделирования не сохранён.

## Финальная техническая проверка комбинации

Воспроизведение CPU-проверок из корня checkout: python3 tests/dsv41/yoco_pipeline15_test.py; затем аналогично три остальных файла ниже. Требуется установленный g++ для компиляции helper harness.

CPU PASS: `tests/dsv41/yoco_pipeline15_test.py`, `yoco_pipeline10_test.py`, `yoco_cone_test.py`, `yoco_pipeline10_cone_test.py`. Проверяют реальные helpers, topology mutations, cutoff, two-slot reuse/ack, nonlinear poison oracle, ненулевую абсолютную позицию, rings/snapshot/taps. Это не полная формальная верификация C++ модели.

HIP gfx906 compilation/link PASS; `git diff --check` PASS. Сборка инкрементальная: новый model.o и отдельный test main, остальные объекты из существующей конфигурации сборки. Команды компиляции записаны в локальном build.log. Compiler: ROCm clang++, -O3 -DNDEBUG -std=c++20 --offload-arch=gfx906, STRATA_HIP_GFX906=1, STRATA_USE_HIP=1, native experts и fused/MMQ prefill. USE_PROF_API=1 присутствует в базовой сборке, но runtime profiling выключен. Инструкции чистой сборки: [AMD_HIP.md](AMD_HIP.md); конкретная локальная incremental-команда извлекалась через ninja -t commands, а не подменяла toolchain. CUDA/SYCL/Metal не собирались.

Allocation gate PASS на обоих GPU. Runtime: `YOCO_CONE active=1 n=20160 cut_k=2 cut_r=4052 last_rows=6720`; `yoco=20 keep=2668`. Запрос завершился, не завис на handoff. После restore/160 teacher-forced токенов pos20320, ok=1. Полный prefill vector129280 и 11 continuation vectors конечны. Generated response завершился естественно после 8 токенов; это не тест скорости длинной генерации.

Сравнение сохранённых полных векторов:
| Сравнение | Prefill max_abs / RMS | Максимум по 12 vectors max_abs / RMS | Совпадений top1 |
| --- | --- | --- | ---: |
| Combined vs прежний 15/25 control1 | 1,626498 / 0,343929 | 2,946554 / 0,664739 | 12/12 |
| Combined vs прежний 10/30 rectangular | 0,816830 / 0,153380 | 1,779536 / 0,290929 | 12/12 |

Это не bit equality и не доказательство отсутствия ухудшения качества. Top1 на 12 точках и нормальный короткий ответ дают только ограниченный технический gate. Качество длинных задач/корпуса не измерено. В старой 10/30 серии два полностью восстановленных 15/25 контроля были byte-identical, а 10/30 отличался; поэтому все различия нельзя списывать на шум.

## Другие проверенные направления

### Exact reconstruction cache

Коммит `571bf4b586044cb257c9989ed569b67755dde169`: i_wq_b/wo_a/wo_b кэшировались на время слоя с прежней reconstruction/dtype. Первый вариант освобождал 144MiB backing перед ростом до 154MiB, после чего headroom admission отказывал. Исправление оставляло первый backing, выбирало целиком помещающиеся матрицы largest-first со stable ties, очищало views/ready на каждом слое. Cap192MiB, reserve256MiB; это admission, не гарантия runtime free.

Оба GPU реально использовали partial/reused cache144MiB. Одна пара дала+1,689%; logits не bit-exact, контрольный разброс в этой серии не отделён. Первые off/on попытки без allocation вообще не считались тестом ускорения. Кэш не включён в финальную комбинацию.

### Uniform F16 routed backend

Коммит `8f8b4107185af9124ab845278da12fb41864ae64`. На production shapes в harness70/70 случаев retained-row invariance прошли побитово. Для выбранных reference normalized RMS light0,009911–0,011282, F16 0,000690–0,000749. Это synthetic/kernel evidence, не качество всей модели.

Цена слишком велика: group1 GU+down0,08538→0,75502ms (8,84× медленнее), group16 0,282251→1,331411ms (4,72×). Не включён. Удаление Q8 activation само по себе не даёт бесплатного ускорения.

### Генерация и DSpark

Умеренный synthetic prompt347 токенов; max512 в HTTP фактически распался из-за thinking budget: 256 generated за 7,538 с (33,96tok/s), затем 239 за 8,597 с (27,80tok/s), плюс 17 inserted wrap tokens. Нельзя назвать это 512 непрерывно сгенерированными токенами.

Первые 88 rounds: 2,909 output tokens/round, 244offered/168accepted=68,85%. Draft1,120258 с, verify6,289067 с (83,55%), other0,117673 с. Block5 уже вычисляет drafts до cap. Greedy принимает target argmax prefix; stochastic path использует min(1,p/q), residual max(0,p−q), rollback/seed. Эта политика не менялась.

Предложенный cutoff0.5 давал лишь условные+1,54/+3,15%; отдельный эксперимент не запускался и в production его нет. Даже полное удаление draft в этой декомпозиции ограничено примерно 17,5%.

### Verify critical path

Коммит меток `9a5e86a1bd886c2b644838cab432fe4eb63f13e0`. Один целевой запрос, 21sampledround послеwarmup: wall1453,36ms, event-chain1412,91ms; GPU0538,13ms, GPU1874,77ms. Runtime здесь forward_serial. Межустройственные event timestamps не вычитались.

Доли: routedGU/down35,39%; QKV/output25,41%; sparse10,51%; index4,35%; router7,20%; shared6,48%; exposedCPUwait1,59%; other6,28%; residual2,78%. Marker overhead ориентировочно 4–7% на sampled/unsampled разныхтокенах, поэтому это диагностические доли, не новый production benchmark. Residual не изолирует H2D.

При verify83,55% ускорение GU/down в 1,3 раза дало бы лишь~7,3% whole-decode. Для+10% потребовалось бы~1,444× GU/down, для+30%~4,555× при прочих равных. CPU lane уже перекрывается GPU-работой; длительность CPU работы нельзя прибавлять к exposed wait. Большого доказанного резерва от «простаивающего CPU» не найдено.

### Persistent group bound

Коммит `d8d26a40de90a921875348b8c2207bcb05d6d2ff`: opt-in ограничение обхода реальным n_groups вместо пустых T*k entries. 192 harness cases, output bits/graph reuse/sentinel PASS. Семь чередующихся пар, 15 повторов; при half-maximum groups компонентный выигрыш 0,8–1,8%, near-full — шум. При ng1 относительный 12–31%, но только 9–37µs.

Histogram реальныхng отсутствовал, поэтому это bracket, не workload-weighted результат. Даже 1,8% типичногоGU/down означало бы лишь~0,53% decode. E2E не запускался, кандидат отвергнут и в комбинацию не входит.

## Где находятся свидетельства

Все каталоги ниже локальные, не публикуются целиком:
- `/var/tmp/strata-analysis-20261027-2101/`: ранние измерения и ограничения step3.
- `/var/tmp/strata-exact-cache-partial-local-1028-evidence/`: cache off/on, allocation, summary.
- `/var/tmp/strata-yoco15-1028-evidence/`: первоначальный 15/25.
- `/var/tmp/strata-yoco-cone-1028-evidence/`: первоначальный cone.
- `/var/tmp/strata-yoco-attn-replay-1029/`, `strata-yoco-moe-replay-1029/`, `strata-yoco-dispatch-replay-1029/`: документы и диагностические milestones; corrected dispatch evidence имеет suffix `-retry-evidence`.
- `/var/tmp/strata-yoco15-trace-1029-evidence/`: trace и conditional balance-model.
- `/var/tmp/strata-yoco10-1029-evidence/`: matched controls,160-token restore и allocation.
- `/var/tmp/strata-uniform-f16-1029/`: standalone F16 probe.
- `/var/tmp/strata-decode-baseline-1029-evidence/`, `strata-verify-phases-1029-evidence/`, `strata-group-bound-1029-evidence/`: decode и отрицательные проверки.
- `/var/tmp/strata-yoco10-cone-1029-evidence/`: final build.log, cpu-tests.log, combined-allocation.json, test-summary.json, numeric-comparison.json, deployment.json и rollback.py.

## Воспроизведение и рабочий запуск

Ветка: `mi50-yoco10-cone-combined-1029`. Source worktree: `/var/tmp/strata-yoco10-cone-1029`.
Чистый binary: `build-cloud/strata-dsv41`, SHA256:
`f6db7a472bb534907a36ad68f365961b4964982790bbce5c586d23dea1b2e14b`.
Test binary SHA256:
`47695c71d8f8a8de65c9c85a11d4a6496566f13ad45fb2a069d8194d7786c4f2`.

В штатном config изменены exe и необходимые флаги:
```text
DS4_SPLIT=10,30
DS4_YOCO_PIPELINE10=1
DS4_YOCO=1
DS4_YOCO_CONE=1
DS4_EXACT_F16_CACHE=0
```
PIPELINE15 и KEEP override отсутствуют. UniformF16, cutoff, replay, phase trace, group-bound и profiling не включены. HIPK, DSpark, модель, profile и sampler сохранены. Production и test main различаются, но используют одинаковый model.o. В production не остаётся диагностического dump/restore main.

Штатный запуск использует историческое имя config `/var/tmp/strata-1026/strata-split-01-21.json`: имя НЕ обозначает текущий split. Команда воспроизводится с локальными путями модели:
```sh
cd /var/tmp/strata-1026
/var/tmp/strata-main/.venv/bin/python3 -m serve.server --engine strata \
  --config /var/tmp/strata-1026/strata-split-01-21.json \
  --tokenizer "$TOKENIZER_DIR" --port 8000 --host "$EXISTING_BIND_HOST"
```
Здесь placeholders обозначают уже настроенные tokenizer/bind; сетевые настройки не менялись. Повторно запускать команду поверх действующей службы нельзя. Отдельный systemd unit в этой работе не создавался.

Откат сохраняет прежний 15/25 config/profile/binary в final evidence. Old binary SHA256:
`de859f9609b306c6ec181a381d11071141b4df450e61ff62d7533a6f831fa887`.
Откат выполняется локальным подготовленным скриптом, который проверяет собственный PID/exe и ждёт READY:
```sh
/var/tmp/strata-main/.venv/bin/python3 \
  /var/tmp/strata-yoco10-cone-1029-evidence/rollback.py
```
Этот скрипт не входит в публикацию: он привязан к конкретной службе и локальному состоянию. Старый вариант сохранён только для отката; после успешного deployment комбинации он не обслуживает запросы.

### Подтверждённое состояние после переключения

29 октября 2026, 07:32:22 UTC: чистая production-служба достигла READY. Server PID 241884, engine PID 241893; /proc/engine/exe указывает на указанный выше clean binary, SHA256 совпал. Health HTTP200. Один короткий запрос на арифметику вернул правильный ответ за0,486с. Этот smoke проверяет службу; его короткий prompt не активирует cone. Активация cone подтверждена отдельным завершённым20160-token запуском выше.

Runtime whitelist подтвердил DS4_SPLIT=10,30, PIPELINE10=1, YOCO=1, CONE=1, EXACT_F16_CACHE=0, STRATA_DS4_HIPK=1. KEEP, PIPELINE15, CODEX_AB_CONTROL, PROF, FLOW_STATS, PIPELINE_DIAG и VERIFY_PHASES отсутствуют. Сохранённый штатный config SHA256: 6334a3bae5180a08bb3e3249175b64225fbafe0f1948d37125413657f0b27303. Изменены только exe, log и необходимые env entries. Сервер оставлен на комбинации; новые эксперименты после этого этапа не запускались.

Публикация предназначена для новой ветки airnsk/Strata — mi50-yoco10-cone-combined-1029. Она не меняет main и прежнюю mi50-yoco15-prototype-1028. Код и этот отчёт доступны вместе; сырые evidence остаются на стенде. Более крупные будущие проекты (единый быстрый точный expert backend, изменение меж-GPU расписания, независимая корпусная оценка качества) потребуют отдельной работы; измеренный выигрыш для них не заявляется.
