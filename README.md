# Coopvisuomotor-fNIRS

This dataset contains simultaneous fNIRS hyperscanning and haptic-device behavioral data from 64 healthy adults organized into 32 dyads. Each dyad completed 15 valid trials in each of four fixed-order conditions: leader-follower/carrot (`run-01`), leader-follower/wave (`run-02`), egalitarian/carrot (`run-03`), and egalitarian/wave (`run-04`).

## Organization

The dataset uses a SNIRF-based, BIDS-inspired dyad-level layout. It is not presented as strict BIDS because one fNIRS system stored both participants in a single 52-channel recording and the behavioral data are also dyad-level. Channels 1-26 belong to member A and channels 27-52 to member B. Participant identifiers use the form `dyad-001A` and `dyad-001B`.

- `dyad-[001-032]/nirs/`: four validated SNIRF recordings per dyad.
- `dyad-[001-032]/beh/`: continuous behavioral samples and trial-level summaries for each run.
- `sourcedata/`: native unprocessed `.nirs` recordings corresponding to the released SNIRF files.
- `derivatives/technical-validation/`: machine-readable results reported in the Technical Validation section.
- `code/`: final scripts and parameter files for the three technical-validation analyses.
- `stimuli/`: reference images of the task interfaces.

The aggregate `task-coopvm_events.tsv` contains 1,920 trial onsets (32 dyads x 4 runs x 15 trials). Onsets are derived from the fNIRS trial-start markers, event duration is the fixed 20-s coloring period, and `completion_time_s` records the behavioral completion time. `task-coopvm_channel-pairs.tsv` defines the 26 cross-participant channel pairs used in the WTC analysis.

## fNIRS acquisition

The NirScan continuous-wave system recorded 52 dyadic channels at 11 Hz using wavelengths of 730, 808, and 850 nm. All 128 SNIRF files passed the official `pysnirf2` 0.7.3 validator on 2026-08-04. Source-detector geometry is shared across all released recordings.

Intermediate Homer files used for the QC, HRF, and WTC analyses are not distributed. They can be regenerated from `sourcedata/` using the workflows recorded in `code/preprocessing_hrf_parameters.json` and `code/preprocessing_wtc_parameters.json`.

## Behavioral data

Continuous behavioral files contain trial-indexed samples from both Touch haptic devices at 1000 Hz. They do not contain an absolute clock column; within-trial elapsed time can be reconstructed as the zero-based row index divided by 1000. Trial summaries provide force, position, total scores, and actual completion time.

## Reuse notes

Run order was fixed, so condition effects may be confounded with order or fatigue. Roles were fixed by member label within each collaboration mode. The montage did not include short-separation channels. Broad-region labels describe montage intent, whereas the MNI coordinates and anatomical labels in the channel table report the AtlasViewer output.

## License and citation

Data and documentation are released under CC BY 4.0. Cite the accompanying data descriptor and the dataset DOI: https://doi.org/10.57760/sciencedb.46254.
