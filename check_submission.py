#!/usr/bin/env python3

# Tested with python 3.12
# Requires: numpy, rich, joblib, tqdm, pillow (all can be installed via pip or other package managers)

# USAGE: python check_submission.py PATH IMG_PATH [-q] [-l LIMIT]
# PATH: path to submission directory (if an archive is given instead, it will be extracted and removed!)
# IMG_PATH: path to directory of images to use as basis for checking the presence of all necessary masks in submission
# -q: quick mode, only check file presence, not file contents
# -l: limit the number of files printed in error messages (default 5, 0 will print all)

import argparse
from collections import defaultdict
import contextlib
from functools import partial
import joblib
import numpy as np
import operator as op
from pathlib import Path
from PIL import Image
import re
from rich import print
from rich.prompt import Confirm, Prompt
import shutil  # os.rename and pathlib.Path.rename fail if the destination is on another filesystem so we use shutil where that can happen
from tqdm import tqdm
import os 

def main():
	errors = defaultdict(list)

	# List what directories and files should be present, along with common mistakes that can be automatically fixed (these mistakes will be renamed to the correct name by this script)
	necessary = {
		training: ({
			testing: ({
				pb: (
					_get_files(testing),
					{pb.lower(), pb.upper(), pb.replace("s", "z")}
				)
				for pb in ('Predictions', 'Binarised')
				},
				{testing.lower(), testing.replace("SM", "MS")}
			)
			for testing in ('MOBIUS', 'SMD+SLD', 'Synthetic')
			},
			{training.lower(), training.upper(), training.replace("+", " + "), training.replace("SM", "MS"), re.sub(r'(.*)\+(.*)', r'\2+\1', training)}
		)
		for training in ('Synthetic', 'Mixed')
	}
	
	# Extract archive if necessary
	# The two whiles in this block handle cases where the archive has several extensions (such as .tar.gz) or is compressed multiple times
	while args.submission.is_file():
		print("Extracting archive")
		shutil.unpack_archive(args.submission)
		print("Extraction successful, removing archive")
		args.submission.unlink()
		while not args.submission.exists():
			args.submission = args.submission.with_suffix('')

	if not args.submission.is_dir():
		raise ValueError(f"{args.submission} is not a directory.")

	# Automatically fix single nested directory
	content = list(args.submission.iterdir())
	if len(content) == 1 and content[0].is_dir():
		print("Fixing single nested directory")
		for d in content[0].iterdir():
			d.rename(args.submission/d.name)
		content[0].rmdir()
		if Confirm.ask("Do you wish to rename the root directory?", default=False):
			new_name = Prompt.ask("Please choose a new name", default=content[0].name)
			args.submission = args.submission.rename(args.submission.with_name(new_name))

	# Rename Segmentator
	while args.submission.name.lower() == "segmentator":
		new_name = Prompt.ask(f"Cannot use the name {args.submission.name}, please choose a different one")
		args.submission = args.submission.rename(args.submission.with_name(new_name))

	# Check entire necessary tree structure
	_check(args.submission, necessary, errors)

	warnings = 'pred', 'rgb', 'bin'
	if errors:
		for err, msg in (('dir', "Missing directories"), ('file', "Missing files"), ('jpg', "Files in JPEG format"), ('pred', "Binary prediction files"), ('rgb', "Colour prediction files"), ('bin', "Non-binary masks")):
			if err in errors:
				colour = 'magenta' if err in warnings else 'red'
				e = errors[err]
				if args.limit <= 0 or len(errors) <= args.limit:
					print(f"[{colour}][b]{msg}[/b][/{colour}]: {list(map(str, e))}")
				else:
					print(f"[{colour}][b]{msg}[/b][/{colour}] (listing {args.limit}/{len(e)}): {list(map(str, e[:args.limit]))}")
	else:
		print("Submission [green][b]OK[/b][/green]")


def _get_files(dataset):
	for data in args.data.rglob(dataset):
		if data.is_dir():
			break
		
	else:
		raise ValueError(f"Couldn't find {dataset} in {args.data}")
	files = [f.relative_to(data).with_suffix('.png') for f in data.rglob('*.*')]
	# if evaluation images are in the "Images" subdirectory, we need to remove the subdirectory name from the path 
	files = [Path(f.name) if "Images" in f.parts or "images" in f.parts else f for f in files]
	# skip "masks" directories in submission 
	files = [f for f in files if "Masks" not in f.parts and "masks" not in f.parts]
	return files


def _check(curr_dir, remaining_tree, errors):
	if isinstance(remaining_tree, dict):  # We're in an inner (directory) node
		for child, (child_remaining, common_mistakes) in remaining_tree.items():
			child_dir = curr_dir/child
			if not child_dir.is_dir():
				# Automatically fix "Binarized"->"Binarised", "SMD + SLD"->"SMD+SLD", "SLD+SMD"->"SMD+SLD", "MSD"->"SMD", ...
				# First filter out the correct child name from common mistakes (if it's in there) to avoid unnecessary checks
				for common_mistake in filter(partial(op.ne, child), common_mistakes):
					mistake_dir = curr_dir/common_mistake
					if mistake_dir.is_dir():
						print(f"Renaming {common_mistake} to {child}")
						mistake_dir.rename(child_dir)
						break
				else:
					errors['dir'].append(child_dir.relative_to(args.submission))
					continue
			_check(child_dir, child_remaining, errors)

	else:  # We're in a leaf node (full of files)
		if args.quick:
			for fname in tqdm(remaining_tree, desc=f"Checking files in {curr_dir.relative_to(args.submission)}"):
				
				curr_f = curr_dir/fname
				if not curr_f.is_file():
					errors['file'].append(curr_f.relative_to(args.submission))
		else:
			
			with tqdm_joblib(tqdm(remaining_tree, desc=f"Checking files in {curr_dir.relative_to(args.submission)}")) as data:
				file_errors = filter(None, joblib.Parallel(n_jobs=-1)(
					joblib.delayed(_check_file)(curr_dir/fname) # data if platform.system().lower() != 'windows' else None)
					for fname in data
				))
			for err_type, err_f in file_errors:
				errors[err_type].append(str(err_f.relative_to(args.submission)))


def _check_file(f):
	# Check the required files
	if not f.is_file():
		for ext in ('.jpg', '.jpeg'):
			if f.with_suffix(ext).is_file():
				return 'jpg', f.with_suffix(ext)
		else:
			return 'file', f
	img = Image.open(f)
	arr = np.array(img)
	unique = np.unique(arr)
	if f.parent.name.lower() == 'predictions':
		# Check that predictions aren't binary
		if len(unique) == 2 and (img.mode == '1' or tuple(unique) == (0, 255)):
			return 'pred', f
		# And not RGB
		if arr.ndim > 2 and not all(np.all(pixel == pixel[0]) for row in arr for pixel in row):
			return 'rgb', f
	# Check that binarised masks are binary
	elif f.parent.name.lower() == 'binarised' and (len(unique) > 2 or img.mode != '1' and not all(lambda x: x in {0, 255} for x in unique)):
		return 'bin', f
	return None


# https://stackoverflow.com/a/58936697/5769814 can get called on intermediate results,
# so it ends up occasionally going over 100%. So we use this solution instead.
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	"""
	Context manager to patch joblib to report into tqdm progress bar given as argument.

	From: https://stackoverflow.com/a/61689175/5769814

	Examples
	--------
	.. code-block:: python

		with tqdm_joblib(tqdm(files)) as data:
			Parallel(n_jobs=-1)(
				delayed(process_file)(f)
				for f in data
			)
	"""

	def tqdm_print_progress(self):
		if self.n_completed_tasks > tqdm_object.n:
			n_completed = self.n_completed_tasks - tqdm_object.n
			tqdm_object.update(n=n_completed)

	original_print_progress = joblib.parallel.Parallel.print_progress
	joblib.parallel.Parallel.print_progress = tqdm_print_progress

	try:
		yield tqdm_object
	finally:
		joblib.parallel.Parallel.print_progress = original_print_progress
		tqdm_object.close()


def process_command_line_options():
	ap = argparse.ArgumentParser(description="Copy files selectively to a new directory.")
	ap.add_argument('--submission', '-s', type=Path, default="SSBC_SEG_PREDICTIONS", help="Submission directory or archive", )
	ap.add_argument('--data', '-d', type=Path, default="SSBC_DATASETS_400x300/Evaluation", help="Directory with original image files to check against")
	ap.add_argument('-q', '--quick', action='store_true', help="Don't check file contents")
	ap.add_argument('-l', '--limit', type=int, default=5, help="Max number of files to print in error messages (if 0, will print all)")
	return ap.parse_args()


if __name__ == '__main__':
	args = process_command_line_options()
	main()
