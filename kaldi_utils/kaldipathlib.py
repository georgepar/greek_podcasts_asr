import datetime
import locale
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import click

locale.setlocale(locale.LC_ALL, "C")


def RunKaldiCommand(command, wait=True):
    """Runs commands frequently seen in Kaldi scripts. These are usually a
    sequence of commands connected by pipes, so we use shell=True"""
    # logger.info("Running the command\n{0}".format(command))
    p = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode != 0:
            raise Exception(
                "There was an error while running the command {0}\n------------\n{1}".format(
                    command, stderr
                )
            )
        return stdout, stderr
    else:
        return p


def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}:{minutes}:{seconds}"


def hhmmss(sec):
    return str(convert_timedelta(datetime.timedelta(seconds=sec)))


def parse_key_value_file(_file):
    with open(_file, "r") as fd:
        lines = [ln.strip().split() for ln in fd]
        keys = [ln[0] for ln in lines]
        values = [ln[1:] for ln in lines]
    return keys, values


def write_key_value_file(_dict, out_file):
    with open(out_file, "w") as fd:
        for k, v in _dict.items():
            fd.write(f"{k} {str(v)}\n")


class FileDict(dict):
    __slots__ = ()

    def __init__(self, mapping=(), **kwargs):
        super(FileDict, self).__init__(mapping)
        self.underlying_file = kwargs.get("underlying_file", None)

    @staticmethod
    def _parser(_file):
        keys, values = parse_key_value_file(_file)
        values = [" ".join(v) for v in values]
        return keys, values

    @classmethod
    def from_file(cls, _file):
        k, v = cls._parser(_file)
        inst = cls(zip(k, v), underlying_file=_file)
        return inst

    def write(self, wav_scp):
        write_key_value_file(self, wav_scp)


class WavScp(FileDict):
    @classmethod
    def from_file(cls, _file):
        inst = super(WavScp, cls).from_file(_file)
        inst = inst.try_convert_to_absolute_paths()
        return inst

    def check_pipes(self):
        return any(v.endswith("|") for v in self.values())

    def _dump_pipe(self, k, v, output_wav_folder):
        if not v.endswith(" - |"):
            return k, v
        v = v.rstrip(" - |")
        uttid = k
        utt_wav = f"{output_wav_folder}/{uttid}.wav"
        cmd = v + f" {utt_wav}"
        RunKaldiCommand(cmd)
        return uttid, utt_wav

    def dump_pipes(self, output_wav_folder):
        uttids, utt_wavs = zip(
            *[self._dump_pipe(k, v, output_wav_folder) for k, v in self.items()]
        )
        return uttids, utt_wavs

    def try_convert_to_absolute_paths(self):
        if self.check_pipes():
            return self
        for k in self.keys():
            self[k] = str(Path(self[k]).absolute())
        return self

    def change_wav_path(self, new_path):
        if self.check_pipes():
            raise ValueError(
                f"change_wav_path works when the wav.scp just contains simple paths. "
                f"The provided {self.underlying_file} contains pipes."
            )
        for k in self.keys():
            filename = Path(self[k]).name
            self[k] = str(Path(new_path).absolute() / filename)
        return zip(*self.items())


class Utt2Spk(FileDict):
    pass


class Text(FileDict):
    pass


@dataclass
class Segment:
    st: float
    et: float
    wav_id: str

    def __str__(self):
        return f"{self.wav_id} {self.st} {self.et}"

    def duration(self):
        return self.et - self.st

    @classmethod
    def from_value(cls, v):
        wav_id, st, et = v
        return cls(st=float(st), et=float(et), wav_id=wav_id)


class Segments(FileDict):
    @staticmethod
    def _parser(_file):
        keys, values = parse_key_value_file(_file)
        values = [Segment.from_value(v) for v in values]
        return keys, values

    def duration(self):
        return sum(v.duration() for v in self.values())


def _get_utterance_prefix(original_wav_scp, wav_scp):
    key = list(original_wav_scp.keys())[0]
    other_key = ""
    for k in wav_scp.keys():
        if k.endswith(key):
            other_key = k
            break
    return other_key.rstrip(key)


def mix_wav_scps(original_wav_scp, augmented_wav_scps, output_file=None):
    assert all(
        len(w.keys()) == len(original_wav_scp.keys()) for w in augmented_wav_scps
    ), (
        f"Expecting wav_scps to have the same number of utterances with {original_wav_scp.underlying_file}."
        f"Wav scps provided {[w.underlying_file for w in augmented_wav_scps]}"
    )

    prefixes = [
        _get_utterance_prefix(original_wav_scp, wscp) for wscp in augmented_wav_scps
    ]

    new_wav_scp = {}

    choose_idx = list(range(len(augmented_wav_scps) + 1))

    for k in original_wav_scp.keys():
        choice = random.choice(choose_idx)
        if choice == len(augmented_wav_scps):
            new_wav_scp[k] = original_wav_scp[k]
        else:
            wscp = augmented_wav_scps[choice]
            prfx = prefixes[choice]
            new_wav_scp[k] = wscp[prfx + k]

    new_wav_scp = WavScp(new_wav_scp.items(), underlying_file=output_file)
    if output_file is not None:
        new_wav_scp.write(output_file)
    return new_wav_scp


def mix_kaldi_dirs(original_kaldi_dir, augmented_kaldi_dirs, output_dir):
    original_kaldi_dir = KaldiPath(original_kaldi_dir).validate()
    augmented_kaldi_dirs = [KaldiPath(d).validate() for d in augmented_kaldi_dirs]
    output_dir = original_kaldi_dir.copy(output_dir)
    original_wav_scp = original_kaldi_dir.parse_wav_scp()
    wav_scps = [kd.parse_wav_scp() for kd in augmented_kaldi_dirs]
    out_wav_scp = output_dir.wav_scp
    new_wav_scp = mix_wav_scps(original_wav_scp, wav_scps, output_file=out_wav_scp)
    output_dir.wav_scp = new_wav_scp
    output_dir.validate().fix()
    return output_dir


def fix_kaldi_dir(kaldi_dir):
    subprocess.run(["utils/fix_data_dir.sh", str(kaldi_dir)], check=True)


def validate_kaldi_dir(kaldi_dir):
    subprocess.run(
        ["utils/validate_data_dir.sh", "--no-feats", str(kaldi_dir)], check=True
    )


def extract_segments_data_dir(in_kaldi_dir, out_kaldi_dir, nj=24):
    subprocess.run(
        [
            "utils/data/extract_wav_segments_data_dir.sh",
            "--nj",
            str(nj),
            in_kaldi_dir,
            out_kaldi_dir,
        ],
        check=True,
    )


def combine_kaldi_dirs(dest_dir, kaldi_dirs):
    args = [
        "utils/combine_data.sh",
        str(dest_dir),
    ] + list(map(str, kaldi_dirs))
    subprocess.run(args, check=True)


class KaldiPath:
    def __init__(self, kaldi_dir: Union[str, Path]):
        # super(KaldiPath, self).__init__(str(kaldi_dir))
        self.kaldi_dir = Path(kaldi_dir)
        self.wav_scp = self.kaldi_dir / "wav.scp"
        self.segments = self.kaldi_dir / "segments"
        self.spk2utt = self.kaldi_dir / "spk2utt"
        self.text = self.kaldi_dir / "text"
        self.utt2spk = self.kaldi_dir / "utt2spk"
        self.reco2file_and_channel = self.kaldi_dir / "reco2file_and_channel"

    def parse_wav_scp(self):
        wav_scp = WavScp.from_file(self.wav_scp)
        return wav_scp

    def parse_text(self):
        text = Text.from_file(self.text)
        return text

    def parse_segments(self):
        segments = Segments.from_file(self.segments)
        return segments

    def __str__(self):
        return str(self.kaldi_dir)

    def validate(self):
        validate_kaldi_dir(str(self))
        return self

    def fix(self):
        fix_kaldi_dir(str(self))
        return self

    def copy(self, other_kaldi_dir):
        shutil.copytree(str(self), other_kaldi_dir)
        new_path = KaldiPath(other_kaldi_dir).validate().fix()
        return new_path

    def combine(self, other_kaldi_dirs, dest_dir):
        for _d in other_kaldi_dirs:
            validate_kaldi_dir(str(_d))
            fix_kaldi_dir(str(_d))
        combine_kaldi_dirs(str(dest_dir), [str(self)] + other_kaldi_dirs)
        out = KaldiPath(dest_dir).validate().fix()
        return out


@click.group()
def cli():
    pass


@cli.command()
@click.argument("original_kaldi_dir", type=str, nargs=1)
@click.argument("augmented_kaldi_dirs", type=str, nargs=-1)
@click.argument("output_dir", type=str, nargs=1)
def mix(original_kaldi_dir, augmented_kaldi_dirs, output_dir):
    output_dir = mix_kaldi_dirs(original_kaldi_dir, augmented_kaldi_dirs, output_dir)
    click.secho(
        f"Successfully mixed {original_kaldi_dir} and {augmented_kaldi_dirs} into {output_dir}",
        fg="blue",
        bold=True,
    )


@cli.command()
@click.argument("kaldi_dirs", type=str, nargs=-1)
@click.argument("output_dir", type=str, nargs=1)
def combine(kaldi_dirs, output_dir):
    combine_kaldi_dirs(output_dir, [d for d in kaldi_dirs])
    KaldiPath(output_dir).validate().fix()
    click.secho(
        f"Successfully combined {kaldi_dirs} into {output_dir}",
        fg="blue",
        bold=True,
    )


@cli.command()
@click.argument("src_kaldi_dir", nargs=1)
@click.argument("dst_kaldi_dir", nargs=1)
def copy(src_kaldi_dir, dst_kaldi_dir):
    kd = KaldiPath(src_kaldi_dir)
    _ = kd.copy(dst_kaldi_dir)
    click.secho(
        f"Successfully copied {src_kaldi_dir} into {dst_kaldi_dir}",
        fg="blue",
        bold=True,
    )


@cli.command()
@click.argument("kaldi_dir", nargs=1)
def duration(kaldi_dir):
    kd = KaldiPath(kaldi_dir)
    segments = kd.parse_segments()
    dur = segments.duration()
    click.secho(f"Total Duration {hhmmss(dur)}", fg="magenta", bold=True)


@cli.command()
@click.option("--nj", type=int, default=24)
@click.argument("src_kaldi_dir", nargs=1)
@click.argument("dst_kaldi_dir", nargs=1)
def extract_segments(src_kaldi_dir, dst_kaldi_dir, nj=24):
    extract_segments_data_dir(src_kaldi_dir, dst_kaldi_dir, nj=nj)
    click.secho(
        f"Successfully extracted wav segments from {src_kaldi_dir}. New kaldi dir: {dst_kaldi_dir}",
        fg="blue",
        bold=True,
    )


@cli.command()
@click.argument("src_kaldi_dir", nargs=1)
@click.argument("dst_kaldi_dir", nargs=1)
@click.argument("wav_folder", type=str, nargs=1)
def execute_wav_scp(src_kaldi_dir, dst_kaldi_dir, wav_folder):
    os.makedirs(wav_folder)
    kd = KaldiPath(src_kaldi_dir)
    kdst_kaldi_dir = kd.copy(dst_kaldi_dir)
    wav_scp = kd.parse_wav_scp()
    uttids, utt_wavs = wav_scp.dump_pipes(wav_folder)
    new_wav_scp = WavScp(zip(uttids, utt_wavs)).try_convert_to_absolute_paths()
    new_wav_scp.write(kdst_kaldi_dir.wav_scp)
    click.secho(
        f"Successfully dumped piped commands in wav.scp from {src_kaldi_dir} into {wav_folder}. New kaldi dir: {dst_kaldi_dir}",
        fg="blue",
        bold=True,
    )


@cli.command(
    help="Change path in wav.scp. Useful when handling a wav.scp with absolute paths"
)
@click.option("--inplace", is_flag=True, show_default=True, default=False)
@click.argument("wav_scp", nargs=1)
@click.argument("new_path", nargs=1)
def change_wav_scp(wav_scp, new_path, inplace=False):
    wscp = WavScp.from_file(wav_scp)
    uttids, utt_wavs = wscp.change_wav_path(new_path)
    out_wav_scp = wav_scp if inplace else wav_scp + ".new"
    new_wav_scp = WavScp(zip(uttids, utt_wavs)).try_convert_to_absolute_paths()
    new_wav_scp.write(out_wav_scp)
    click.secho(
        f"Written wav.scp with new base path in {out_wav_scp}",
        fg="blue",
        bold=True,
    )


if __name__ == "__main__":
    cli()
