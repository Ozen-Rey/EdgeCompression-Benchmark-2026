# Innesti per la tesi: routing content-aware

## Sommario

Il caso di studio viene inoltre esteso a una selezione content-aware nel dominio immagine. Rispetto a una baseline globale robusta, una policy source-aware riduce il regret medio di circa l'87%, mentre un classificatore source-agnostic basato su sole feature metadata-only mantiene una riduzione del 75.7% anche nel protocollo leave-one-dataset-out. Il predictor non sostituisce la logica R-D-E, ma propone una configurazione che viene accettata solo se ammissibile rispetto ai vincoli di qualita, hardware e sistema.

## Introduzione

Il contributo applicativo del router non si limita alla selezione system-aware. Nel dominio immagine, la tesi valuta anche se il contenuto possa anticipare quale configurazione R-D-E sara piu vicina all'oracle per immagine. A questo scopo vengono confrontate una baseline globale robusta, una policy source-aware, un oracle non deployable e classificatori source-agnostic validati con protocolli leave-one-image-out e leave-one-dataset-out. La proposta content-aware rimane subordinata ai vincoli del router: il classificatore suggerisce un candidato, ma la decisione finale viene presa sul pool ammissibile e con lo stesso costo di ranking usato dalla logica R-D-E/system-aware.

## Fusione content-aware e system-aware

La policy predittiva agisce come suggerimento consultivo. Dopo i filtri di disponibilita, capacita, qualita e sistema, il router costruisce il pool ammissibile \(A_{adm}\). Il costo di ranking resta:

```tex
\[
J_{rank}(a)=
\begin{cases}
J_{RDE}(a), & \text{senza penalita di sistema applicata},\\
J_{RDE}(a)+\lambda_{sys}P_{sys}(a), & \text{con penalita di sistema applicata}.
\end{cases}
\]
```

La selezione finale diventa:

```tex
\[
a_{sel}=
\begin{cases}
\hat{a}_{cont}(x), &
\text{se } \hat{a}_{cont}(x)\in A_{adm}
\text{ ed e competitiva secondo } J_{rank},\\
\arg\min_{a\in A_{adm}} J_{rank}(a), & \text{altrimenti}.
\end{cases}
\]
```

In questo modo il classificatore non bypassa il quality guard, cioe il vincolo robusto di qualita, ne le penalita di sistema.

## Discussione dei risultati

Poiche la validazione e condotta su 96 immagini, i risultati vanno interpretati come evidenza prototipale e non come stima definitiva della generalizzazione su distribuzioni industriali molto piu ampie.

L'ablazione non ha lo scopo di stabilire una superiorita definitiva di kNN su modelli supervisionati piu complessi, ma di verificare che il beneficio content-aware non dipenda da una scelta arbitraria del classifier.

## Conclusioni e limiti

Il router content-aware e implementato come prototipo deployable e validato offline sui punti operativi misurati. Il limite non e quindi l'assenza del predictor, ma l'assenza di una validazione online end-to-end su nuovi contenuti, nuove piattaforme e workload reali.

La validazione del routing content-aware resta offline rispetto al database R-D-E misurato. Il prototipo implementa gia una policy predittiva leggera, ma non e ancora stato validato in un deployment online completo con misure energetiche aggiornate sulla piattaforma corrente.
