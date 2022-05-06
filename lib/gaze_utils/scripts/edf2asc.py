import subprocess
import argparse
from glob import glob
import os
import os.path

EDF2ASC_OPTIONS = [
    '-sg',   # outputs sample GAZE data if present (default)
    '-nv',   # hide viewer commands
    '-nmsg', # blocks message event output
    '-t',    # use only tabs as delimiters
    ]

def edf_to_asc(edf_file: str, out_dir: str, options: list) -> str:
    """
    Calls the edf2asc command and saves it.

    :param edf_file: Path to edf file.
    :param out_dir: Where to place the asc file.
    :returns: Path to asc file.
    """
    assert(edf_file[-4:] == ".edf")
    if out_dir is None:
        out_file = f'{edf_file[:-4]}.asc'
    else:
        out_file = os.path.join(out_dir, os.path.basename(edf_file)[:-4] + '.asc')
    if out_dir is not None:
        subprocess.run(["edf2asc", edf_file, out_file] + options)
    else:
        subprocess.run(["edf2asc", edf_file] + options)
    return out_file


if __name__ == '__main__':
    print("WARNING: Requires 'edf2asc' to installed!")

    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_file", type=str, default=None)
    parser.add_argument("--edf_glob_pattern", type=str)
    parser.add_argument("--asc_out_dir", type=str, default=None)
    parser.add_argument("--options", type=str, default=EDF2ASC_OPTIONS)
    args = vars(parser.parse_args())

    if args['edf_file'] is not None:
        print(edf_to_asc(args['edf_file'], args['asc_out_dir'], args['options']))

    elif args['edf_glob_pattern'] is not None:
        for path in glob(args['edf_glob_pattern']):
            edf_to_asc(path, args['asc_out_dir'], args['options'])

