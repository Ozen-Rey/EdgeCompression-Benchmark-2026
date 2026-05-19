# Innesti per la tesi: routing R-D-E predittivo e neural-inclusive

Questo documento contiene testo pronto per l'integrazione nel manoscritto,
in particolare nel Capitolo 5 e nel Capitolo 6. Il contenuto non introduce
nuovi esperimenti: riorganizza e contestualizza i risultati v0.43.0-v0.43.4
gia prodotti dagli artefatti offline del router.

## Capitolo 5 - Routing R-D-E predittivo

### Benchmark R-D-E come base decisionale

La metodologia R-D-E non ha solo una funzione descrittiva. Il benchmark
definisce un insieme di punti operativi misurati, ciascuno associato a rate,
qualita e costo energetico, e fornisce quindi la base quantitativa per una
decisione vincolata. Il router non sceglie un codec in astratto, ma seleziona
una configurazione all'interno del pool ammissibile, dopo l'applicazione dei
vincoli di qualita, disponibilita, capacita e profilo operativo. In questo
senso, la stima del costo R-D-E e il ranking multi-obiettivo costituiscono il
meccanismo che trasforma il benchmark in una politica di selezione.

Questa impostazione e rilevante anche per l'estensione predittiva. Il
predittore puo suggerire un candidato, ma la decisione finale resta vincolata
al quality guard, al pool ammissibile e alla ranking key attiva. Il routing
predittivo va quindi interpretato come un'estensione controllata della logica
R-D-E, non come una sostituzione del criterio di selezione.

### Classic-only content-aware routing come ablation controllata

Il primo caso predittivo e una ablation classic-only sul dominio immagine. Il
pool considerato e limitato alle configurazioni classiche JPEG q=85, JXL d=1.0
e HEVC crf=15. Questa scelta non deve essere presentata come risultato finale
sul routing R-D-E completo, ma come esperimento controllato: dato un pool
classico e deployable, si valuta se una policy leggera basata sulla sorgente o
sui soli metadati dell'immagine possa ridurre il regret rispetto a una baseline
globale robusta.

Nel benchmark considerato, la distribuzione oracle classic-only e sbilanciata:
JPEG q=85 compare in 63 casi, JXL d=1.0 in 30 casi e HEVC crf=15 in 3 casi.
HEVC e quindi troppo raro per sostenere claim strutturali sulla frontiera
decisionale. L'analisi principale dell'ablazione riguarda il problema JPEG/JXL,
mentre i casi HEVC restano utili come esempi qualitativi e come controllo di
classe minoritaria.

### Policy comparison e incertezza bootstrap

La Tabella seguente riassume la comparazione classic-only. Il regret e
calcolato rispetto all'oracle per immagine dello stesso pool classico; le
riduzioni sono riportate rispetto alla robust global baseline.

```tex
\begin{table}[t]
\centering
\caption{Confronto classic-only delle policy di routing content-aware.}
\label{tab:classic-only-policy-comparison}
\begin{tabular}{l l r r l}
\toprule
Policy & Protocollo & Mean regret & Relative reduction & Bootstrap 95\% CI \\
\midrule
Robust global baseline & global & 0.09047 & 0.0\% & [0.0\%, 0.0\%] \\
Source-aware majority & leave-one-out & 0.01177 & 87.0\% & [81.2\%, 92.1\%] \\
kNN metadata-only & LOIO & 0.01238 & 86.3\% & [80.5\%, 91.3\%] \\
kNN metadata-only & LODO & 0.02197 & 75.7\% & [69.0\%, 82.2\%] \\
Per-image oracle & oracle & 0.00000 & 100.0\% & [100.0\%, 100.0\%] \\
\bottomrule
\end{tabular}
\end{table}
```

La policy source-aware funziona bene quando la sorgente o il batch di
provenienza sono noti, perche sfrutta una regolarita di dominio gia osservata
nel benchmark. Il risultato metadata-only in protocollo LOIO indica invece che
feature leggere, senza accesso ai pixel e senza etichetta di sorgente,
catturano comunque un segnale predittivo. Il protocollo LODO e il test piu
severo, perche esclude l'intero dataset target dal training e valuta la
generalizzazione cross-dataset. La presenza di intervalli bootstrap rende il
claim piu robusto rispetto alla dimensione del corpus, pari a N=96 immagini,
ma non trasforma il risultato in una stima universalistica su tutte le immagini
naturali. Il claim resta relativo a un benchmark multi-source eterogeneo.

### Dal predittore alla decisione vincolata
\label{sec:router-predictor-decisione-vincolata}

Il predittore e consultivo: suggerisce una configurazione candidata, mentre il
router decide. La proposta viene accettata solo se appartiene al pool
ammissibile e se resta competitiva rispetto alla ranking key attiva. I vincoli
di qualita, il safe pool e il ranking R-D-E rimangono quindi vincolanti. Se la
proposta non supera il quality guard, non e presente nel pool ammissibile o non
e competitiva rispetto al costo di ranking, il router ricade sulla normale
selezione R-D-E all'interno del pool ammissibile.

Formalmente, indicata con \(\hat{a}_{cont}(x)\) la proposta content-aware e
con \(A_{adm}\) il pool ammissibile, la selezione puo essere descritta come:

```tex
\[
a_{sel} =
\begin{cases}
\hat{a}_{cont}(x), &
\text{se } \hat{a}_{cont}(x)\in A_{adm}
\text{ ed e competitiva secondo } J_{rank},\\
\arg\min_{a\in A_{adm}} J_{rank}(a), &
\text{altrimenti.}
\end{cases}
\]
```

Nel caso source-aware Tecnick, la policy suggerisce JPEG q=85. Il router
accetta il suggerimento perche la configurazione passa il quality guard,
appartiene al safe pool ed e competitiva rispetto alla ranking key attiva
\(J_{RDE}\), con \(J_{RDE}=0.19003\) e vincolo di qualita minimo almeno pari a
80. La decisione e quindi registrata come content policy accepted. Questo
esempio mostra che il predittore non aggira il router: propone un candidato che
viene poi validato dai vincoli operativi.

### Interpretabilita del predittore metadata-only
\label{sec:metadata-only-predictor-interpretability}

L'audit di interpretabilita classic-only mostra che il guadagno del kNN
metadata-only non deriva da una black box opaca. Sul problema JPEG/JXL, il suo
comportamento e quasi completamente approssimabile da poche soglie su feature
di metadato, principalmente megapixels e aspect ratio.

Il surrogate tree di profondita 1 raggiunge una fidelity di circa 86.0\%
rispetto alle predizioni kNN. A profondita 2 la fidelity sale a circa 98.9\%,
mentre profondita 3 e 4 non producono miglioramenti sostanziali. Questo
risultato suggerisce che, nel benchmark considerato, la frontiera appresa dal
kNN sul sottoproblema JPEG/JXL e semplice. L'attribution conferma che
megapixels e la feature dominante, aspect_ratio ha un ruolo secondario, mentre
orientation e resolution class hanno contributi quasi nulli. Le interazioni
logistiche vanno trattate come esplorative e descrittive, non come evidenza
conclusiva di causalita o significativita statistica.

Il risultato e utile proprio per la sua semplicita: il fast path metadata-only
e leggero, interpretabile e coerente con una politica deployable che evita la
scansione completa dei pixel quando non necessaria.

### Audit neural-inclusive dello spazio R-D-E
\label{sec:neural-inclusive-rde-audit}

L'ablazione classic-only non e sufficiente per discutere il routing R-D-E
completo. Il pool full include sia codec classici sia codec neurali, e consente
di chiedere in quali regioni dello spazio rate-distortion-energy i codec
neurali diventino oracle-optimal. L'audit neural-inclusive deve quindi essere
letto come complemento dell'ablazione, non come sua sostituzione.

L'audit neural-inclusive mostra che il vantaggio dei codec neurali non puo
essere valutato lungo il solo asse del bitrate. Nel benchmark considerato, i
codec neurali diventano oracle-optimal soprattutto nel profilo
bandwidth-limited, mentre nei profili energy-limited e quality-first il pool
classico rimane quasi equivalente al full pool. Quando un codec neurale vince,
la riduzione di rate va valutata insieme alla penalita energetica, che puo
essere elevata. Il claim centrale e quindi prudente: i codec neurali non sono
globalmente migliori, ma occupano una regione specifica dello spazio R-D-E.
Questo non implica dominanza universale ne rende obsoleti i codec classici.

### Valutazione predittiva neural-inclusive
\label{sec:neural-inclusive-predictive-evaluation}

L'audit oracle e un limite teorico: calcola quale configurazione sarebbe
migliore dopo aver osservato le misure R-D-E disponibili per ciascuna immagine.
La valutazione predittiva neural-inclusive risponde a una domanda diversa. Il
router sceglie senza vedere le misure R-D-E della target image; solo dopo la
scelta si calcola il regret rispetto all'oracle full-pool. Questa distinzione
e essenziale per evitare leakage metodologico.

La Tabella seguente riassume il caso bandwidth-limited con vincolo
PSNR \(\geq 30\). I campi non disponibili sono marcati come n.d., senza
inferire valori mancanti.

```tex
\begin{table}[t]
\centering
\caption{Valutazione predittiva neural-inclusive nel profilo bandwidth-limited con PSNR $\geq 30$.}
\label{tab:neural-inclusive-predictive-router}
\begin{tabular}{l l r r r r r}
\toprule
Policy & Protocollo & Mean regret & Reduction vs global & Neural predicted & Neural precision & Neural recall \\
\midrule
Robust global baseline & global & 0.0360 & 0\% & 0\% & n.d. & n.d. \\
Source-aware full-pool & LOIO & 0.0096 & 73\% & n.d. & 0.67 & 0.50 \\
kNN metadata full-pool & LOIO & 0.0097 & 73\% & 30\% & 0.59 & 0.53 \\
kNN metadata classic-only & LOIO & 0.0114 & 68\% & 0\% & n.d. & n.d. \\
kNN metadata full-pool & LODO & n.d. & 64\% & n.d. & n.d. & n.d. \\
kNN metadata classic-only & LODO & n.d. & 50\% & 0\% & n.d. & n.d. \\
\bottomrule
\end{tabular}
\end{table}
```

Nel profilo bandwidth-limited con PSNR almeno pari a 30, la robust global
baseline ottiene mean regret 0.0360 e non predice codec neurali, mentre
l'oracle full-pool seleziona una famiglia neurale nel 33\% dei casi. Le policy
predittive full-pool riducono il regret a circa 0.0096-0.0097 in LOIO, con una
riduzione di circa 73\% rispetto alla baseline globale. Il kNN metadata
full-pool predice codec neurali in circa il 30\% dei casi, con precision 0.59 e
recall 0.53 rispetto alla famiglia oracle; la policy source-aware full-pool
raggiunge precision 0.67 e recall 0.50.

Il confronto tra kNN full-pool e kNN classic-only quantifica il costo operativo
dell'esclusione dei neurali: nel protocollo LODO, la riduzione del regret passa
da circa 50\% a circa 64\%, indicando che il pool neural-inclusive conserva
valore anche sotto generalizzazione cross-dataset. Il vantaggio
neural-inclusive non e quindi solo teorico: anche un router predittivo leggero
riesce a recuperare parte del vantaggio dei neurali senza leakage. Questo
risultato resta circoscritto al benchmark considerato e non implica che i codec
neurali siano sempre preferibili.

### Best practice operative

I risultati suggeriscono alcune pratiche operative per l'uso del routing
predittivo. Primo, le policy predittive devono restare subordinate ai vincoli
del router: quality guard, pool ammissibile e ranking multi-obiettivo devono
essere applicati dopo la proposta del predittore. Secondo, i protocolli di
valutazione devono separare chiaramente la scelta predittiva dalla misura del
regret: le misure R-D-E della target image possono essere usate per valutare la
decisione, non per sceglierla. Terzo, il confronto tra pool classic-only e pool
neural-inclusive deve essere presentato come confronto tra regimi operativi,
non come gerarchia assoluta tra famiglie di codec.

### Sintesi del Capitolo 5

Nel complesso, la metodologia R-D-E produce decisioni predittive e vincolate,
non solo descrizioni di benchmark. Il router non assume che i codec neurali o
classici siano sempre migliori: identifica il regime operativo in cui ciascun
paradigma puo risultare conveniente. Nel benchmark considerato, le policy
classic-only mostrano che sorgente e metadati leggeri contengono segnale
predittivo; l'audit neural-inclusive mostra che i neurali occupano soprattutto
una regione bandwidth-limited dello spazio R-D-E; la valutazione predittiva
neural-inclusive indica che una policy metadata-only puo recuperare parte di
questo vantaggio senza usare misure R-D-E della target image in fase di scelta.
Le decisioni restano rese sicure da quality guard, fallback e ranking
multi-obiettivo. Il dominio audio rimane piu difficile, soprattutto per
l'eterogeneita delle metriche e per la minore immediatezza nel confrontare
distorsione percettiva, rate ed energia entro una singola ranking key.

## Capitolo 6 - Artefatti di analisi e osservabilita

### Artefatti di analisi predittiva e osservabilita
\label{sec:artefatti-analisi-predittiva-osservabilita}

Gli artefatti v0.43.0-v0.43.4 non modificano il runtime router. Sono moduli
offline e read-only che consumano risultati e report esistenti per produrre
tabelle, spiegazioni e audit metodologici. La loro funzione nel Capitolo 6 non
e ricostruire una cronologia di release, ma chiarire la struttura concettuale
della pipeline di analisi predittiva.

```tex
\begin{table}[t]
\centering
\caption{Artefatti offline per analisi predittiva e osservabilita del router.}
\label{tab:predictive-analysis-observability-artifacts}
\begin{tabular}{p{0.25\linewidth} p{0.22\linewidth} p{0.22\linewidth} p{0.21\linewidth}}
\toprule
Modulo & Input & Output & Ruolo \\
\midrule
\texttt{policy\_comparison.py} &
Decisioni e regret delle policy classic-only &
Tabelle CSV/JSON con mean regret, reduction e CI bootstrap &
Confronta le policy content-aware e quantifica l'incertezza su N=96. \\
\texttt{decision\_explanation.py} &
Report JSON di una decisione router &
Spiegazione Markdown/JSON della decisione &
Rende osservabile la separazione tra suggerimento predittivo, ammissibilita e ranking. \\
\texttt{content\_predictor\_interpretability.py} &
Oracle classic-only, feature metadata-only e decisioni kNN &
Audit JSON/CSV/TXT con class balance, surrogate tree e attribution &
Verifica se il fast path metadata-only e interpretabile e segnala lo sbilanciamento HEVC. \\
\texttt{neural\_inclusive\_oracle.py} &
Benchmark full-pool con classici e neurali &
Oracle per profilo, floor e pool; confronto classic-only vs full-pool &
Misura la regione teorica in cui i neurali diventano oracle-optimal. \\
\texttt{neural\_inclusive\_predictive\_router.py} &
Benchmark full-pool e label oracle su fold di training &
Decisioni e summary predittivi LOIO/LODO con regret e metriche neural-family &
Valuta se policy leggere recuperano parte del vantaggio full-pool senza leakage. \\
\bottomrule
\end{tabular}
\end{table}
```

Tutti questi moduli sono offline e read-only: non eseguono codec, non
rigenerano benchmark, non cambiano il ranking score, non modificano il pool
runtime e non alterano lo schema operativo del report router. Il loro ruolo e
rendere tracciabile la catena inferenziale: dal benchmark R-D-E, alla policy
predittiva, alla decisione vincolata, fino al regret calcolato ex post.

### Future work
\label{sec:future-work-routing-predittivo}

Rimangono aperte alcune direzioni che estendono la pipeline senza confonderla
con gli artefatti gia implementati. Una prima direzione e l'uncertainty-aware
routing, con confidence gating completo: il predittore dovrebbe esporre una
stima di incertezza utilizzabile dal router per decidere se accettare la
proposta, allargare il pool o ricadere sulla baseline. Una seconda direzione e
la drift detection, necessaria per riconoscere quando il corpus operativo si
allontana dal benchmark o dai profili di calibrazione. Collegato a questo
punto, l'active re-benchmarking permetterebbe di aggiornare selettivamente i
punti R-D-E quando drift o obsolescenza rendono insufficienti le misure
esistenti.

Altre estensioni riguardano la consapevolezza delle risorse e del contesto:
RAM/VRAM-aware routing, context inheritance o workload memory per batch
correlati, un orchestratore multi-domain che tratti immagine, video e audio
entro una stessa logica decisionale, e uno scale-up su corpus piu ampi. Non
vanno invece elencati come future work il feedback logging, la proposta e
validazione di calibrazione, l'external codec registry, la policy comparison,
la decision explanation, l'audit oracle neural-inclusive o la valutazione
predittiva neural-inclusive, poiche questi elementi sono gia presenti come
artefatti offline o componenti documentate del router.
