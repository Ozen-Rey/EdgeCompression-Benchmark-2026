import argparse

from src.router.core.profiles import available_profiles


def build_router_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype R-D-E router for adaptive codec selection."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Router configuration JSON file. Expanded before normal argument parsing.",
    )

    parser.add_argument("--csv", required=True, help="Path del CSV con i punti R-D-E.")

    parser.add_argument(
        "--calibration-bundle-manifest",
        default=None,
        help=(
            "Explicit calibration bundle manifest. When provided, the router "
            "validates the manifest and uses its calibrated CSV as the R-D-E input."
        ),
    )

    parser.add_argument(
        "--calibration-bundle-validation",
        default=None,
        help=(
            "Optional explicit shadow decision validation report. When provided, "
            "it must be accepted before the calibration bundle can be used."
        ),
    )

    parser.add_argument(
        "--calibration-file",
        default=None,
        help="File JSON di calibrazione locale da applicare ai punti R-D-E.",
    )

    parser.add_argument(
        "--normalization-file",
        default=None,
        help="File JSON con scale di normalizzazione precomputate.",
    )

    parser.add_argument(
        "--normalization-mode",
        default="auto",
        choices=["auto", "runtime", "global", "dataset", "local"],
        help=(
            "Politica di normalizzazione: auto, runtime, global, dataset, local. "
            "runtime usa la normalizzazione calcolata al volo; global/dataset/local "
            "richiedono --normalization-file."
        ),
    )

    parser.add_argument(
        "--previous-decision-receipt",
        default=None,
        help=(
            "Optional explicit previous decision receipt/router report used only "
            "to audit normalization comparability in the output report."
        ),
    )

    parser.add_argument(
        "--input",
        default=None,
        help="File di input da usare per generare un piano di esecuzione.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="File di output desiderato per il piano di esecuzione.",
    )

    parser.add_argument(
        "--generate-command",
        action="store_true",
        help="Genera un execution plan per il codec selezionato.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Esegue direttamente il comando generato se il piano è eseguibile.",
    )

    parser.add_argument(
        "--domain",
        default="image",
        choices=["image", "audio", "video"],
        help="Dominio multimediale.",
    )

    parser.add_argument(
        "--profile",
        default="balanced",
        choices=available_profiles(),
        help="Profilo operativo da usare se --all-profiles non è attivo.",
    )

    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Esegue il router su tutti i profili disponibili e genera un summary CSV.",
    )

    parser.add_argument(
        "--auto-weights",
        action="store_true",
        help="Calcola automaticamente i pesi R-D-E dal contesto operativo.",
    )

    parser.add_argument(
        "--power-mode",
        choices=["ac", "battery", "unknown"],
        default="ac",
        help="Modalità alimentazione usata dalla policy contestuale.",
    )

    parser.add_argument(
        "--battery-percent",
        type=float,
        default=None,
        help="Percentuale batteria usata dalla policy contestuale.",
    )

    parser.add_argument(
        "--thermal-state",
        choices=["nominal", "warm", "hot", "critical"],
        default="nominal",
        help="Stato termico usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--network-profile",
        choices=["normal", "limited", "very-limited"],
        default="normal",
        help="Profilo rete usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--quality-target",
        choices=["preview", "normal", "high", "very-high"],
        default="normal",
        help="Target qualità usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--quality-thresholds-file",
        default="configs/quality_thresholds.json",
        help="File JSON con soglie qualità domain-specific.",
    )

    parser.add_argument(
        "--codec-registry-file",
        default=None,
        help="External codec/backend registry JSON file.",
    )

    parser.add_argument(
        "--external-codec-manifest",
        action="append",
        default=[],
        help=(
            "Explicit external codec router manifest. Repeat to load multiple "
            "validated external R-D-E exports. No automatic discovery is performed."
        ),
    )

    parser.add_argument(
        "--system-load",
        choices=["normal", "high", "very-high"],
        default="normal",
        help="Carico sistema usato dalla policy contestuale.",
    )

    parser.add_argument(
        "--aggregate-by-config",
        action="store_true",
        help="Aggrega le righe per codec+config usando la media di rate, qualità ed energia.",
    )

    parser.add_argument(
        "--normalization-scope",
        choices=["global", "filtered"],
        default="global",
        help=(
            "global = normalizza sui punti prima dei filtri codec; "
            "filtered = normalizza solo sui punti rimasti dopo i filtri."
        ),
    )

    parser.add_argument(
        "--available-codecs",
        default=None,
        help="Lista separata da virgole dei codec disponibili. Esempio: JPEG,JXL,HEVC",
    )

    parser.add_argument(
        "--exclude-codecs",
        default=None,
        help="Lista separata da virgole dei codec da escludere. Esempio: DCAE,JPEG_AI",
    )

    parser.add_argument(
        "--exclude-neural",
        action="store_true",
        help="Esclude codec neurali o basati su modelli appresi.",
    )

    parser.add_argument(
        "--system-aware",
        action="store_true",
        help="Usa il profilo del sistema reale per filtrare automaticamente il pool ammissibile.",
    )

    parser.add_argument(
        "--system-features",
        action="store_true",
        help="Extract cheap system-aware features and include them in the report.",
    )

    parser.add_argument(
        "--system-probe-level",
        default="basic",
        choices=["basic", "gpu", "full"],
        help="System feature probe level: basic, gpu, or full.",
    )

    parser.add_argument(
        "--system-feature-cache-ttl-s",
        type=float,
        default=5.0,
        help="Cache TTL in seconds for system feature probes.",
    )

    parser.add_argument(
        "--system-feature-cpu-interval-s",
        type=float,
        default=0.0,
        help="psutil CPU sampling interval for system feature extraction.",
    )

    parser.add_argument(
        "--system-policy",
        action="store_true",
        help="Build a system-aware policy from extracted system features.",
    )

    parser.add_argument(
        "--system-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records suggested changes; apply uses adjusted weights.",
    )

    parser.add_argument(
        "--system-policy-simulate",
        default=None,
        help=(
            "Comma-separated simulated system classes, e.g. "
            "'battery=critical,cpu=busy,memory=constrained'. "
            "Overrides measured classes for system-policy evaluation."
        ),
    )

    parser.add_argument(
        "--system-penalty",
        action="store_true",
        help="Compute a codec/backend system penalty from resource profiles.",
    )

    parser.add_argument(
        "--system-penalty-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records J_total; apply ranks by J_total and applies hard exclusions.",
    )

    parser.add_argument(
        "--system-penalty-lambda",
        type=float,
        default=0.25,
        help="Weight of the system penalty term in J_total.",
    )

    parser.add_argument(
        "--system-penalty-weights-file",
        default=None,
        help="Optional JSON file with configurable system penalty coefficients.",
    )

    parser.add_argument(
        "--content-policy",
        action="store_true",
        help="Enable source-aware content policy reporting.",
    )

    parser.add_argument(
        "--content-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content policy mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-policy-rules-file",
        default=None,
        help="CSV rules file produced by content metadata policy evaluation.",
    )

    parser.add_argument(
        "--content-policy-key",
        default="dataset",
        help="Metadata key used by the content policy rules, e.g. dataset/source.",
    )

    parser.add_argument(
        "--content-source",
        default=None,
        help="Optional homogeneous content source/dataset label, e.g. tecnick, kodak, clic2020.",
    )

    parser.add_argument(
        "--content-source-filter",
        action="store_true",
        help="Filter the benchmark candidate pool using the provided content source/context.",
    )

    parser.add_argument(
        "--content-filter-column",
        default=None,
        help="CSV/raw column used for source-conditioned filtering. Defaults to --content-policy-key.",
    )

    parser.add_argument(
        "--content-filter-value",
        default=None,
        help="Value used for source-conditioned filtering. Defaults to --content-source.",
    )

    parser.add_argument(
        "--content-classifier",
        action="store_true",
        help="Enable source-agnostic content classifier reporting.",
    )

    parser.add_argument(
        "--content-classifier-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content classifier mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-classifier-config",
        default=None,
        help="JSON config for the source-agnostic content classifier.",
    )

    parser.add_argument(
        "--content-classifier-image",
        default=None,
        help="Optional image path used to extract source-agnostic content classifier features.",
    )

    parser.add_argument(
        "--content-classifier-width",
        type=int,
        default=None,
        help="Optional image width used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--content-classifier-height",
        type=int,
        default=None,
        help="Optional image height used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--capability-aware",
        action="store_true",
        help="Filtra i codec usando il registry dei requisiti hardware/software.",
    )

    parser.add_argument(
        "--strict-executables",
        action="store_true",
        help="Se attivo, esclude i codec i cui eseguibili richiesti non sono nel PATH.",
    )

    parser.add_argument(
        "--simulate-no-cuda",
        action="store_true",
        help="Debug: simula assenza di CUDA per testare il filtro system-aware.",
    )

    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Attiva guardia qualità robusta: usa p10 se non specificato e floor minimo 60.",
    )

    parser.add_argument(
        "--quality-constraint-stat",
        choices=["mean", "p25", "p10", "min"],
        default=None,
        help="Statistica usata come vincolo duro di qualità.",
    )

    parser.add_argument(
        "--quality-floor",
        type=float,
        default=None,
        help="Soglia minima assoluta di qualità accettabile.",
    )

    parser.add_argument(
        "--near-quality-floor",
        type=float,
        default=None,
        help="Soglia qualità quasi-usabile per fallback degradato.",
    )

    parser.add_argument(
        "--allow-degraded-fallback",
        action="store_true",
        help="Permette fallback degradato se nessun punto supera la soglia sicura.",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Qualità minima ammissibile. Se assente, usa quella del profilo/policy.",
    )

    parser.add_argument("--max-rate", type=float, default=None, help="Rate massimo ammissibile.")
    parser.add_argument("--max-energy", type=float, default=None, help="Energia massima ammissibile.")
    parser.add_argument("--max-time-ms", type=float, default=None, help="Tempo massimo ammissibile in millisecondi.")
    parser.add_argument(
        "--strict-time",
        action="store_true",
        help=(
            "Se usato con --max-time-ms, richiede che tutti i punti candidati "
            "abbiano time_ms disponibile."
        ),
    )

    parser.add_argument("--wE", "--w-e", dest="wE", type=float, default=None, help="Peso energia custom.")
    parser.add_argument("--wR", "--w-r", dest="wR", type=float, default=None, help="Peso rate custom.")
    parser.add_argument("--wD", "--w-d", dest="wD", type=float, default=None, help="Peso distorsione custom.")

    parser.add_argument("--codec-col", default=None)
    parser.add_argument("--config-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--quality-col", default=None)
    parser.add_argument("--quality-metric", default=None)
    parser.add_argument("--energy-col", default=None)
    parser.add_argument("--time-col", default=None)

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Numero di configurazioni migliori da salvare nel report JSON.",
    )

    parser.add_argument(
        "--export-topk",
        action="store_true",
        help="Esporta anche i top-k candidati in CSV.",
    )

    parser.add_argument(
        "--feedback-out",
        default="results/routing_context/online_feedback.csv",
        help=(
            "Append-only CSV for observed execution feedback. "
            "Written only when --execute is used."
        ),
    )

    parser.add_argument(
        "--out",
        default="results/routing/router_decision_report.json",
        help="Path del report JSON quando si usa un solo profilo.",
    )

    parser.add_argument(
        "--out-dir",
        default="results/routing",
        help="Cartella di output quando si usa --all-profiles.",
    )

    parser.add_argument(
        "--summary-out",
        default=None,
        help="Path del summary CSV. Se assente, usa results/routing/router_summary.csv.",
    )

    return parser
