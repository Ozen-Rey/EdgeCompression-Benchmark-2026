from pathlib import Path

from src.router.analysis.content_oracle_analysis import (
    analyze_content_oracle,
    load_oracle_rows,
)


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "content_oracle_analysis"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_content_oracle_analysis_detects_per_image_oracle_diversity():
    csv_path = _tmp_path("content_diversity.csv")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,image,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,img1,CodecQuality,mode=q,1.0,95.0,100.0,100.0",
                "A,img1,CodecEnergy,mode=e,1.2,85.0,1.0,20.0",
                "A,img2,CodecQuality,mode=q,1.0,85.0,100.0,100.0",
                "A,img2,CodecEnergy,mode=e,1.2,95.0,1.0,20.0",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_oracle_rows(
        str(csv_path),
        dataset_col="dataset",
        image_col="image",
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    analysis = analyze_content_oracle(
        rows,
        quality_floor=80.0,
        w_r=0.2,
        w_e=0.2,
        w_d=0.6,
        global_coverage_floor=1.0,
    )

    oracle_pairs = {
        (row["oracle_codec"], row["oracle_config"])
        for row in analysis["by_image"]
    }

    assert len(oracle_pairs) == 2

    summary = {
        (row["section"], row["key"]): row["value"]
        for row in analysis["summary"]
    }

    assert summary[("summary", "num_images_total")] == 2
    assert summary[("summary", "num_images_analyzed")] == 2
    assert summary[("oracle_diversity", "num_distinct_oracle_configs")] == 2


def test_content_oracle_analysis_uses_coverage_constrained_global_baseline():
    csv_path = _tmp_path("content_global_infeasible.csv")

    csv_path.write_text(
        "\n".join(
            [
                "dataset,image,codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "A,img1,GlobalLikely,mode=g,1.0,90.0,1.0,10.0",
                "A,img1,Fallback,mode=f,1.5,85.0,2.0,10.0",
                "A,img2,GlobalLikely,mode=g,1.0,60.0,1.0,10.0",
                "A,img2,Fallback,mode=f,1.5,85.0,2.0,10.0",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_oracle_rows(
        str(csv_path),
        dataset_col="dataset",
        image_col="image",
        codec_col="codec",
        config_col="param",
        rate_col="bpp",
        quality_col="ssimulacra2",
        energy_col="energy_per_image_j",
        time_col="time_ms",
    )

    analysis = analyze_content_oracle(
        rows,
        quality_floor=80.0,
        w_r=0.2,
        w_e=0.2,
        w_d=0.6,
        global_coverage_floor=1.0,
    )

    summary = {
        (row["section"], row["key"]): row["value"]
        for row in analysis["summary"]
    }

    assert summary[("global_best", "codec")] == "Fallback"
    assert summary[("global_best", "config")] == "mode=f"
    assert summary[("global_best", "coverage_rate")] == 1.0
    assert summary[("summary", "num_global_infeasible_images")] == 0
