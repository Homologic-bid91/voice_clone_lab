"""Configuration loading and per-voice path derivation.

`config/default.yaml` is the single source of truth. Data and output paths are
never stored in the YAML — they derive from the speaker name via
:meth:`Config.paths_for`. CLI flags override config values at call time.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "default.yaml"


def project_path(value: str | Path) -> Path:
    """Resolve a path relative to the project root; absolute paths pass through."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def rel_to_root(path: str | Path) -> str:
    """Render a path relative to the project root for portable artifact metadata."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class ProjectConfig:
    speaker: str = "default"
    sample_rate: int = 24000
    device: str = "cuda:0"


@dataclass
class QwenConfig:
    repo_url: str = "https://github.com/QwenLM/Qwen3-TTS.git"
    repo_commit: str = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
    repo_path: str = "third_party/Qwen3-TTS"
    init_model_path: str = "models/Qwen3-TTS-12Hz-1.7B-Base"
    init_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    attn_implementation: str = "auto"  # auto | flash_attention_2 | sdpa


@dataclass
class TrainingConfig:
    batch_size: int = 8
    lr: float = 2e-6
    epochs: int = 5


@dataclass
class GenerationConfig:
    checkpoint: str | None = None
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    language: str = "Auto"


@dataclass
class AudioConfig:
    min_chunk_seconds: float = 3.0
    max_chunk_seconds: float = 12.0
    target_chunk_seconds: float = 8.0
    reference_min_seconds: float = 3.0
    reference_ideal_min_seconds: float = 5.0
    reference_ideal_max_seconds: float = 15.0
    silence_top_db: float = 35.0
    pad_seconds: float = 0.12
    min_rms_dbfs: float = -42.0
    min_snr_db: float = 8.0
    vad_aggressiveness: int = 2
    clean_padding_ms: int = 180
    clean_join_silence_ms: int = 180


@dataclass
class ASRConfig:
    backend: str = "auto"
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"


@dataclass
class SystemConfig:
    min_free_gb: int = 80


@dataclass
class VoicePaths:
    """Every data/output location for one speaker."""

    speaker: str
    voice_dir: Path
    raw_dir: Path
    extracted_audio: Path
    cleaned_audio: Path
    chunks_dir: Path
    chunk_metadata: Path
    transcripts_jsonl: Path
    review_tsv: Path
    reference_audio: Path
    dataset_dir: Path
    train_raw_jsonl: Path
    train_with_codes_jsonl: Path
    dataset_stats: Path
    checkpoints_dir: Path
    generated_dir: Path


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config from YAML. An explicit path that is missing is an error."""
        if path is not None:
            cfg_path = Path(path)
            if not cfg_path.exists():
                raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
        else:
            cfg_path = DEFAULT_CONFIG_PATH
            if not cfg_path.exists():
                return cls()
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Build a Config, tolerating unknown sections/keys (forward compat)."""
        cfg = cls()
        sections = {f.name for f in dataclasses.fields(cls)}
        for section, values in (data or {}).items():
            if section not in sections or not isinstance(values, dict):
                continue
            obj = getattr(cfg, section)
            known = {f.name for f in dataclasses.fields(obj)}
            for key, value in values.items():
                if key in known:
                    setattr(obj, key, value)
        return cfg

    def paths_for(self, speaker: str | None = None) -> VoicePaths:
        """Derive all data/output paths for a speaker (default: project.speaker)."""
        speaker = speaker or self.project.speaker
        voice_dir = ROOT / "data" / "voices" / speaker
        dataset_dir = voice_dir / "dataset"
        return VoicePaths(
            speaker=speaker,
            voice_dir=voice_dir,
            raw_dir=voice_dir / "raw",
            extracted_audio=voice_dir / "extracted" / f"{speaker}.wav",
            cleaned_audio=voice_dir / "cleaned" / f"{speaker}_clean.wav",
            chunks_dir=voice_dir / "chunks",
            chunk_metadata=voice_dir / "chunks" / "metadata.json",
            transcripts_jsonl=voice_dir / "transcripts" / "transcripts.jsonl",
            review_tsv=voice_dir / "transcripts" / "transcripts_review.tsv",
            reference_audio=voice_dir / "reference" / "ref.wav",
            dataset_dir=dataset_dir,
            train_raw_jsonl=dataset_dir / "train_raw.jsonl",
            train_with_codes_jsonl=dataset_dir / "train_with_codes.jsonl",
            dataset_stats=dataset_dir / "dataset_stats.json",
            checkpoints_dir=ROOT / "outputs" / "checkpoints" / speaker,
            generated_dir=ROOT / "outputs" / "generated" / speaker,
        )

    def qwen_repo_path(self) -> Path:
        return project_path(self.qwen.repo_path)

    def qwen_finetuning_path(self) -> Path:
        return self.qwen_repo_path() / "finetuning"

    def init_model_path(self) -> Path:
        return project_path(self.qwen.init_model_path)
